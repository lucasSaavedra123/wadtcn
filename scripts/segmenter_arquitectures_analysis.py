from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import f1_score
from scipy.signal import find_peaks

from Trajectory import Trajectory
from DataSimulation import AndiDataSimulation, Andi2ndDataSimulation
from PredictiveModel.LSTMTheoreticalModelSegmentClassification1 import LSTMTheoreticalModelSegmentClassification1
from PredictiveModel.LSTMTheoreticalModelSegmentClassification2 import LSTMTheoreticalModelSegmentClassification2
from PredictiveModel.LSTMTheoreticalModelSwitchTimeAndExponent import LSTMTheoreticalModelSwitchTimeAndExponent
from PredictiveModel.WavenetTCNSingleLevelAlphaPredicter import WavenetTCNSingleLevelAlphaPredicter
from PredictiveModel.WavenetTCNMultiTaskClassifierSingleLevelPredicter import WavenetTCNMultiTaskClassifierSingleLevelPredicter
from PredictiveModel.WavenetTCNSingleLevelChangePointPredicter import WavenetTCNSingleLevelChangePointPredicter
from andi_datasets.utils_challenge import single_changepoint_error, label_continuous_to_list, segment_property_errors



FROM_ANDI_2 = True

if not FROM_ANDI_2:
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

    results = {
        'tcn': { 'alpha': None, 'model': None},
        'lstm': { 'c1': None, 'c2': None, 'time_exponent': None }
    }

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

    trajectories = AndiDataSimulation().simulate_segmentated_trajectories(12_500, 200, 200)
else:
    networks = {
        'tcn': {
            'd': WavenetTCNSingleLevelAlphaPredicter(200,None,simulator=Andi2ndDataSimulation),
            'alpha': WavenetTCNSingleLevelAlphaPredicter(200,None,simulator=Andi2ndDataSimulation),
            'model': WavenetTCNMultiTaskClassifierSingleLevelPredicter(200,None,simulator=Andi2ndDataSimulation),
            'cp': WavenetTCNSingleLevelChangePointPredicter(200,None,simulator=Andi2ndDataSimulation)
        },
    }

    results = {'tcn': { 'd': None, 'alpha': None, 'model': None, 'cp': None}}

    trajectories = Andi2ndDataSimulation().simulate_phenomenological_trajectories_for_classification_training(12_500, 200, 200, True, 'val',enable_parallelism=True)

    for i, t in enumerate(trajectories):
        sigma = np.random.uniform(0,2)

        labels = np.zeros((t.length, 2))
        labels[:,0] = list(t.info['alpha_t'])
        labels[:,1] = list(t.info['d_t'])
        gt_cps = label_continuous_to_list(labels)[0]
        trajectories[i] = Trajectory(
            x=t.get_x(),
            y=t.get_y(),
            noise_x=np.random.randn(t.length) * sigma,
            noise_y=np.random.randn(t.length) * sigma,
            info={
                'd_t':list(t.info['d_t']),
                'alpha_t':list(t.info['alpha_t']),
                'state_t':list(t.info['state_t']),
                'cps': gt_cps
            }
        )

#Load models
for work_label in networks:
    for net_label in networks[work_label]:
        if net_label == 'cp' and not FROM_ANDI_2:
            networks[work_label][net_label].load_as_file(selected_name='wavenet_changepoint_detector_200_200.0_andi_with_weighted_bce_999.h5')
        else:
            networks[work_label][net_label].load_as_file()
#Predict
for work_label in networks:
    for net_label in networks[work_label]:
        if net_label == 'cp':
            results[work_label][net_label] = networks[work_label][net_label].predict(trajectories, apply_threshold=False)
        else:
            results[work_label][net_label] = networks[work_label][net_label].predict(trajectories)

if not FROM_ANDI_2:
    for t_i in range(len(trajectories)):
        alphas = results['tcn']['alpha'][t_i,:,0]
        models = np.argmax(results['tcn']['model'][t_i,:],axis=1)

        tcp_cp = np.argmax(results['tcn']['cp'][t_i,:,0])
        if tcp_cp == 0:
            tcp_cp = 1
        elif tcp_cp == 199:
            tcp_cp = 198

        lstm_cp = results['lstm']['time_exponent'][1][t_i]
        real_cp = trajectories[t_i].info['change_point_time']

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
else:
    tp_rmses, jaccards = [], []
    for t_i in range(len(trajectories)):
        trajectory = trajectories[t_i]
        preds_cps = find_peaks(results['tcn']['cp'][t_i,:,0], height=0.8464712)[0].tolist() + [trajectory.length]

        plt.plot(results['tcn']['cp'][t_i,:,0])
        for c in trajectory.info['cps']:
            plt.axvline(c, color='black', linewidth=2)
        for c in preds_cps:
            plt.axvline(c, color='red')
        plt.show()

        tp_rmse, jaccard = single_changepoint_error(trajectory.info['cps'], preds_cps, threshold=10)
        tp_rmses.append(tp_rmse)
        jaccards.append(jaccard)

    print(np.mean(tp_rmses), np.mean(jaccards))