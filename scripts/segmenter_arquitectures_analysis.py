from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import f1_score

from DataSimulation import AndiDataSimulation
from PredictiveModel.LSTMTheoreticalModelSegmentClassification1 import LSTMTheoreticalModelSegmentClassification1
from PredictiveModel.LSTMTheoreticalModelSegmentClassification2 import LSTMTheoreticalModelSegmentClassification2
from PredictiveModel.LSTMTheoreticalModelSwitchTimeAndExponent import LSTMTheoreticalModelSwitchTimeAndExponent
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter

networks = {
    'tcn': {
        'alpha': WavenetTCNSingleLevelAlphaPredicter(200,200,simulator=AndiDataSimulation),
        'model': WavenetTCNMultiTaskClassifierSingleLevelPredicter(200,200,simulator=AndiDataSimulation),
        'cp': WavenetTCNSingleLevelChangePointPredicter(200,200,simulator=AndiDataSimulation)
    },
    'lstm': {
        'c1': LSTMTheoreticalModelSegmentClassification1(200,200,simulator=AndiDataSimulation),
        'c2': LSTMTheoreticalModelSegmentClassification2(200,200,simulator=AndiDataSimulation),
        'time_exponent': LSTMTheoreticalModelSwitchTimeAndExponent(200,200,simulator=AndiDataSimulation),
    }
}

trajectories = AndiDataSimulation().simulate_segmentated_trajectories(12_500, 200, 200)

results = {
    'tcn': { 'alpha': None, 'model': None},
    'lstm': { 'c1': None, 'c2': None, 'time_exponent': None }
}

#Load models
for work_label in networks:
    for net_label in networks[work_label]:
        if net_label == 'cp':
            networks[work_label][net_label].load_as_file(selected_name='wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_50.h5')
        else:
            networks[work_label][net_label].load_as_file()
#Predict
for work_label in networks:
    for net_label in networks[work_label]:
        results[work_label][net_label] = networks[work_label][net_label].predict(trajectories)
#Generate metrics
metrics = {
    'distance_of_real_cp' : {
        'lstm': [],
        'tcn': [],
        'real': [],
    },
    'first_alpha': {
        'lstm': [],
        'tcn': [],
        'real': [],
        'cp': [],
    },
    'second_alpha': {
        'lstm': [],
        'tcn': [],
        'real': [],
        'cp': [],
    },
    'first_segment_state': {
        'lstm': defaultdict(lambda:{'real': [], 'predicted': []}),
        'tcn': defaultdict(lambda:{'real': [], 'predicted': []}),
    },
    'second_segment_state': {
        'lstm': defaultdict(lambda:{'real': [], 'predicted': []}),
        'tcn': defaultdict(lambda:{'real': [], 'predicted': []}),
    }
}

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

for t_i in range(len(trajectories)):
    alphas = results['tcn']['alpha'][t_i,:,0]
    models = np.argmax(results['tcn']['model'][t_i,:],axis=1)

    #tcp_cp = np.mean(np.where(results['tcn']['cp'][t_i,:,0]==1)[0])
    
    cps = np.where(results['tcn']['cp'][t_i,:,0]==1)[0]

    if len(cps) > 1:
        #density = scipy.stats.gaussian_kde(cps)
        #x = np.arange(0,200, 0.1)
        #y = density(x)
        #tcp_cp = x[np.argmax(y)]
        tcp_cp = np.mean(cps)
    elif len(cps) == 1:
        tcp_cp = cps[0]
    else:
        tcp_cp = np.argmax(networks['tcn']['cp'].predict([trajectories[t_i]], apply_threshold=False)[0,:,0])
    tcp_cp = int(tcp_cp)

    lstm_cp = results['lstm']['time_exponent'][1][t_i]
    real_cp = trajectories[t_i].info['change_point_time']
    """
    fig,ax = plt.subplots(2,1)
    raw_pred = networks['tcn']['cp'].predict([trajectories[t_i]], apply_threshold=False)[0,:,0]
    ax[0].plot(raw_pred)
    ax[0].plot(moving_average(raw_pred,n=5))
    ax[0].axvline(trajectories[t_i].info['change_point_time'], color='black')
    ax[0].axvline(tcp_cp, color='red')
    ax[0].set_ylim([-0.1,1.1])

    ax[1].plot(results['tcn']['cp'][t_i,:,0])
    ax[1].plot(moving_average(results['tcn']['cp'][t_i,:,0],n=5))
    ax[1].axvline(trajectories[t_i].info['change_point_time'], color='black')
    ax[1].axvline(tcp_cp, color='red')
    ax[1].set_ylim([-0.1,1.1])

    plt.show()
    """

    tcn_first_segment_model_tcp = stats.mode(models[:tcp_cp]).mode
    tcn_second_segment_model_tcp = stats.mode(models[tcp_cp:]).mode
    lstm_first_segment_model_tcp = np.argmax(results['lstm']['c1'][t_i])
    lstm_second_segment_model_tcp = np.argmax(results['lstm']['c2'][t_i])

    tcn_first_segment_alpha_tcp = np.mean(alphas[:tcp_cp]) * 2
    tcn_second_segment_alpha_tcp = np.mean(alphas[tcp_cp:]) * 2
    lstm_first_segment_alpha_tcp = results['lstm']['time_exponent'][0][0][t_i]
    lstm_second_segment_alpha_tcp = results['lstm']['time_exponent'][0][1][t_i]

    #sns.kdeplot(np.where(results['tcn']['cp'][t_i,:,0]==1)[0])
    #plt.axvline(real_cp)
    #plt.show()

    metrics['distance_of_real_cp']['lstm'].append(lstm_cp)
    metrics['distance_of_real_cp']['tcn'].append(tcp_cp)
    metrics['distance_of_real_cp']['real'].append(real_cp)

    metrics['first_alpha']['lstm'].append(lstm_first_segment_alpha_tcp)
    metrics['first_alpha']['tcn'].append(tcn_first_segment_alpha_tcp)
    metrics['first_alpha']['real'].append(trajectories[t_i].info['alpha_first_segment'])
    metrics['first_alpha']['cp'].append(real_cp)

    metrics['second_alpha']['lstm'].append(lstm_second_segment_alpha_tcp)
    metrics['second_alpha']['tcn'].append(tcn_second_segment_alpha_tcp)
    metrics['second_alpha']['real'].append(trajectories[t_i].info['alpha_second_segment'])
    metrics['second_alpha']['cp'].append(real_cp)

    metrics['first_segment_state']['lstm'][real_cp]['predicted'].append(lstm_first_segment_model_tcp)
    metrics['first_segment_state']['lstm'][real_cp]['real'].append(trajectories[t_i].info['model_first_segment'])
    metrics['second_segment_state']['lstm'][real_cp]['predicted'].append(lstm_second_segment_model_tcp)
    metrics['second_segment_state']['lstm'][real_cp]['real'].append(trajectories[t_i].info['model_second_segment'])

    metrics['first_segment_state']['tcn'][real_cp]['predicted'].append(tcn_first_segment_model_tcp)
    metrics['first_segment_state']['tcn'][real_cp]['real'].append(trajectories[t_i].info['model_first_segment'])
    metrics['second_segment_state']['tcn'][real_cp]['predicted'].append(tcn_second_segment_model_tcp)
    metrics['second_segment_state']['tcn'][real_cp]['real'].append(trajectories[t_i].info['model_second_segment'])

fig,ax = plt.subplots(1,3)

df = pd.DataFrame(metrics['distance_of_real_cp'])
df['lstm'] = (df['lstm'] - df['real'])**2
df['tcn'] = (df['tcn'] - df['real'])**2
df = df.groupby(pd.cut(df['real'],np.arange(0,200+20,20))).mean()#df.groupby('real').mean()
df['lstm'] = np.sqrt(df['lstm'])
df['tcn'] = np.sqrt(df['tcn'])
df.to_csv('distance_of_real_cp.csv')

ax[0].plot([i.right for i in df.index.to_list()], df['tcn'], label='tcn', color='blue')
ax[0].plot([i.right for i in df.index.to_list()], df['lstm'], label='lstm', color='red')
ax[0].legend()

df_1_lstm = {'cp': [], 'f1': []}
df_2_lstm = {'cp': [], 'f1': []}
df_1_tcn = {'cp': [], 'f1': []}
df_2_tcn = {'cp': [], 'f1': []}

for cp in metrics['first_segment_state']['lstm']:
    metrics['first_segment_state']['lstm'][cp] = f1_score(
        metrics['first_segment_state']['lstm'][cp]['real'],
        metrics['first_segment_state']['lstm'][cp]['predicted'],
        average='micro'
    )

    df_1_lstm['cp'].append(cp)
    df_1_lstm['f1'].append(metrics['first_segment_state']['lstm'][cp])

for cp in metrics['first_segment_state']['tcn']:
    metrics['first_segment_state']['tcn'][cp] = f1_score(
        metrics['first_segment_state']['tcn'][cp]['real'],
        metrics['first_segment_state']['tcn'][cp]['predicted'],
        average='micro'
    )

    df_1_tcn['cp'].append(cp)
    df_1_tcn['f1'].append(metrics['first_segment_state']['tcn'][cp])

for cp in metrics['second_segment_state']['lstm']:
    metrics['second_segment_state']['lstm'][cp] = f1_score(
        metrics['second_segment_state']['lstm'][cp]['real'],
        metrics['second_segment_state']['lstm'][cp]['predicted'],
        average='micro'
    )

    df_2_lstm['cp'].append(cp)
    df_2_lstm['f1'].append(metrics['second_segment_state']['lstm'][cp])

for cp in metrics['second_segment_state']['tcn']:
    metrics['second_segment_state']['tcn'][cp] = f1_score(
        metrics['second_segment_state']['tcn'][cp]['real'],
        metrics['second_segment_state']['tcn'][cp]['predicted'],
        average='micro'
    )

    df_2_tcn['cp'].append(cp)
    df_2_tcn['f1'].append(metrics['second_segment_state']['tcn'][cp])

df_1_lstm = pd.DataFrame(df_1_lstm).sort_values('cp')
df_1_lstm = df_1_lstm.groupby(pd.cut(df_1_lstm['cp'],np.arange(0,200+20,20))).mean()

df_2_lstm = pd.DataFrame(df_2_lstm).sort_values('cp')
df_2_lstm = df_2_lstm.groupby(pd.cut(df_2_lstm['cp'],np.arange(0,200+20,20))).mean()

df_1_tcn = pd.DataFrame(df_1_tcn).sort_values('cp')
df_1_tcn = df_1_tcn.groupby(pd.cut(df_1_tcn['cp'],np.arange(0,200+20,20))).mean()

df_2_tcn = pd.DataFrame(df_2_tcn).sort_values('cp')
df_2_tcn = df_2_tcn.groupby(pd.cut(df_2_tcn['cp'],np.arange(0,200+20,20))).mean()

df_1_lstm.to_csv('classification_cp_lstm_1.csv')
df_2_lstm.to_csv('classification_cp_lstm_2.csv')
df_1_tcn.to_csv('classification_cp_tcn_1.csv')
df_2_tcn.to_csv('classification_cp_tcn_2.csv')

ax[1].plot([i.right for i in df_1_tcn.index.to_list()], df_1_tcn['f1'], label='tcn', color='blue')
ax[1].plot([i.right for i in df_1_lstm.index.to_list()], df_1_lstm['f1'], label='lstm', color='red')
ax[1].plot([i.right for i in df_2_tcn.index.to_list()], df_2_tcn['f1'], label='tcn', color='blue', linestyle='dashed')
ax[1].plot([i.right for i in df_2_lstm.index.to_list()], df_2_lstm['f1'], label='lstm', color='red', linestyle='dashed')
ax[1].legend()

df = pd.DataFrame(metrics['first_alpha'])
df['lstm'] = (df['lstm'] - df['real']).abs()
df['tcn'] = (df['tcn'] - df['real']).abs()
df = df.groupby(pd.cut(df['cp'],np.arange(0,200+20,20))).mean()#df.groupby('cp').mean()
df.to_csv('first_alpha.csv')

ax[2].plot([i.right for i in df.index.to_list()], df['tcn'], label='tcn', color='blue')
ax[2].plot([i.right for i in df.index.to_list()], df['lstm'], label='lstm', color='red')
ax[2].legend()

df = pd.DataFrame(metrics['second_alpha'])
df['lstm'] = (df['lstm'] - df['real']).abs()
df['tcn'] = (df['tcn'] - df['real']).abs()
df = df.groupby(pd.cut(df['cp'],np.arange(0,200+20,20))).mean()#df.groupby('cp').mean()
df.to_csv('second_alpha.csv')

ax[2].plot([i.right for i in df.index.to_list()], df['tcn'], label='tcn', color='blue', linestyle='dashed')
ax[2].plot([i.right for i in df.index.to_list()], df['lstm'], label='lstm', color='red', linestyle='dashed')
ax[2].legend()

plt.show()
