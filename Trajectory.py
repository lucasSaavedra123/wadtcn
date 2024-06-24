import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import chi2, bootstrap
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import ruptures as rpt
from mongoengine import Document, FloatField, ListField, DictField, BooleanField
from andi_datasets.datasets_challenge import _defaults_andi2
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull
import scipy.stats as st
import matplotlib.animation as animation
from collections import defaultdict
import moviepy.editor as mp
from moviepy.video.fx.all import crop
from moviepy.editor import *

#Example about how to read trajectories from .mat
"""
from scipy.io import loadmat
from Trajectory import Trajectory
mat_data = loadmat('data/all_tracks_thunder_localizer.mat')
# Orden en la struct [BTX|mAb] [CDx|Control|CDx-Chol]
dataset = []
# Add each label and condition to the dataset
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][0][0]})
dataset.append({'label': 'BTX',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][0][1]})
dataset.append({'label': 'BTX',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][0][2]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx',
                'tracks': mat_data['tracks'][1][0]})
dataset.append({'label': 'mAb',
                'exp_cond': 'Control',
                'tracks': mat_data['tracks'][1][1]})
dataset.append({'label': 'mAb',
                'exp_cond': 'CDx-Chol',
                'tracks': mat_data['tracks'][1][2]})
for data in dataset:
    trajectories = Trajectory.from_mat_dataset(data['tracks'], data['label'], data['exp_cond'])
    for trajectory in trajectories:
        if not trajectory.is_immobile(1.8):
            trajectory.save()
"""


def turning_angles(length, x, y, steps_lag=1, normalized=False):
    #if length/steps_lag <= 1:
    #    return []

    X = np.zeros((length,2))
    X[:,0] = x
    X[:,1] = y
    X = X[::steps_lag,:]
    length = X.shape[0]

    steps_lag = 1
    U_t = X[:length-steps_lag]
    U_t_plus_delta = X[np.arange(0, length-steps_lag)+steps_lag]

    V_t = U_t_plus_delta - U_t
    V_t_plus_delta = V_t[np.arange(0,len(V_t)-steps_lag)+steps_lag]
    V_t = V_t[:len(V_t)-steps_lag]

    A = np.sum((V_t_plus_delta * V_t), axis=1)
    B = np.linalg.norm(V_t, axis=1) * np.linalg.norm(V_t_plus_delta, axis=1)

    #Some B values could be 0. These are removed
    values_to_keep = np.where(B != 0)
    A = A[values_to_keep]
    B = B[values_to_keep]

    angles = np.clip(A/B, -1, 1)

    if not normalized:
        angles = np.rad2deg(np.arccos(angles))

    return angles.tolist()

"""
This method is a Array-Oriented Python implementation of a similar algorithm proposed in the referenced
paper to how direction change in time.

Taylor, R. W., Holler, C., Mahmoodabadi, R. G., Küppers, M., Dastjerdi, H. M., Zaburdaev, V., . . . Sandoghdar, V. (2020). 
High-Precision Protein-Tracking With Interferometric Scattering Microscopy. 
Frontiers in Cell and Developmental Biology, 8. 
https://doi.org/10.3389/fcell.2020.590158
"""
def directional_correlation(length, x, y, steps_lag=1, window_size=9):
    assert window_size % 2 == 1, 'Window size has to be odd'
    angles = turning_angles(length, x, y, steps_lag=steps_lag, normalized=True)
    convolution_result = np.convolve(angles, np.ones(window_size), mode='same')/window_size
    return convolution_result[window_size//2:-window_size//2]

def directional_correlation_segmentation(length, x, y, steps_lag=1, window_size=9, pen=1, jump=1, min_size=3, return_break_points=False):
    result = []
    signal = directional_correlation(length, x, y, window_size=window_size, steps_lag=steps_lag)

    break_points = rpt.Pelt(
        model='l2',
        jump=jump,
        min_size=min_size,
        ).fit_predict(
            signal,
            pen=pen
            )

    initial_index = 0
    for break_point in break_points:
        result.append(np.mean(signal[initial_index:break_point]))
        initial_index = break_point

    assert len(result) == len(break_points)

    if return_break_points:
        return result, break_points
    else:
        return result


class Trajectory(Document):
    x = ListField(required=True)
    y = ListField(required=False)
    z = ListField(required=False)

    t = ListField(required=True)

    noise_x = ListField(required=False)
    noise_y = ListField(required=False)
    noise_z = ListField(required=False)

    noisy = BooleanField(required=True)

    info = DictField(required=False)

    @classmethod
    def from_mat_dataset(cls, dataset, label='no label', experimental_condition='no experimental condition', scale_factor=1000): # With 1000 we convert trajectories steps to nm
        trajectories = []
        number_of_tracks = len(dataset)
        for i in range(number_of_tracks):
            raw_trajectory = dataset[i][0]

            trajectory_time = raw_trajectory[:, 0]
            trayectory_x = raw_trajectory[:, 1] * scale_factor
            trayectory_y = raw_trajectory[:, 2] * scale_factor

            trajectories.append(Trajectory(trayectory_x, trayectory_y, t=trajectory_time, info={"label": label, "experimental_condition": experimental_condition}, noisy=True))

        return trajectories

    @classmethod
    def from_datasets_phenom(cls, trajs, labels):
        trajectories = []

        for traj_index in range(trajs.shape[1]):
            selected_snr = np.random.uniform(0.5,5)
            sigma = np.std(np.append(np.diff(trajs[:,traj_index,0]), np.diff(trajs[:,traj_index,1]))) / selected_snr

            trajectories.append(
                Trajectory(
                    x=trajs[:,traj_index,0],
                    y=trajs[:,traj_index,1],
                    t=np.arange(0, len(trajs[:,traj_index,1])) * 0.1, #This frame rate was obtained in the website of the Andi Challenge
                    noise_x=np.random.randn(trajs.shape[0])*sigma,#_defaults_andi2().sigma_noise,
                    noise_y=np.random.randn(trajs.shape[0])*sigma,#defaults_andi2().sigma_noise,
                    info={
                        'alpha_t': labels[:,traj_index,0],
                        'd_t': labels[:,traj_index,1],
                        'state_t': labels[:,traj_index,2]
                    }
                )
            )

        return trajectories

    @classmethod
    def from_models_phenom(cls, trajs, labels):
        trajectories = []

        for traj_index in range(trajs.shape[1]):
            sigma = 0 #During simulation, noise is not added in trajectories saving. Instead, It is added during training

            trajectories.append(
                Trajectory(
                    x=trajs[:,traj_index,0],
                    y=trajs[:,traj_index,1],
                    t=np.arange(0, len(trajs[:,traj_index,1])) * 0.1, #This frame rate was obtained in the website of the Andi Challenge
                    noise_x=np.random.randn(trajs.shape[0])*sigma,#_defaults_andi2().sigma_noise,
                    noise_y=np.random.randn(trajs.shape[0])*sigma,#_defaults_andi2().sigma_noise,
                    info={
                        'alpha_t': labels[:,traj_index,0],
                        'd_t': labels[:,traj_index,1],
                        'state_t': labels[:,traj_index,2]
                    }
                )
            )

        return trajectories    

    @classmethod
    def from_challenge_phenom_dataset(cls, trajs, labels):
        trajectories = []

        for dataframe, current_labels in zip(trajs, labels):
            for traj_idx in dataframe['traj_idx'].unique():
                traj_dataframe = dataframe[dataframe['traj_idx'] == traj_idx]
                traj_dataframe = traj_dataframe.sort_values('frame')

                traj_labels = [l for l in current_labels if l[0] == traj_idx][0]
                traj_labels = traj_labels[1:]

                d = np.zeros(len(traj_dataframe))
                a = np.zeros(len(traj_dataframe))
                s = np.zeros(len(traj_dataframe))

                from_c = 0
                for label_index in range(0, int(len(traj_labels)/4)):
                    d_i = traj_labels[(label_index*4)]
                    a_i = traj_labels[(label_index*4)+1]
                    s_i = traj_labels[(label_index*4)+2]
                    c_i = traj_labels[(label_index*4)+3]

                    d[from_c:c_i] = d_i
                    a[from_c:c_i] = a_i
                    s[from_c:c_i] = s_i
                    from_c = c_i
                assert len(np.unique(d)) != 3
                assert len(np.unique(a)) != 3
                assert len(np.unique(s)) != 3
                trajectories.append(
                    Trajectory(
                        x=traj_dataframe['x'].tolist(),
                        y=traj_dataframe['y'].tolist(),
                        t=(traj_dataframe['frame'] * 0.1).tolist(), #This frame rate was obtained in the website of the Andi Challenge
                        info={
                            'alpha_t': a.tolist(),
                            'd_t': d.tolist(),
                            'state_t': s.tolist()
                        },
                        noisy=True
                    )
                )

        return trajectories

    @classmethod
    def ensemble_average_mean_square_displacement(cls, trajectories, number_of_points_for_msd=50, bin_width=None, alpha=0.95):
        """
        trajectories = [trajectory for trajectory in trajectories if trajectory.length > number_of_points_for_msd + 1]
        #print("len average ->", np.mean([t.length for t in trajectories]))
        ea_msd = np.zeros((len(trajectories), number_of_points_for_msd))
        mu_t = np.zeros((len(trajectories), number_of_points_for_msd))

        for j_index, trajectory in enumerate(trajectories):
            positions = np.zeros((trajectory.length,2))
            positions[:,0] = trajectory.get_noisy_x()
            positions[:,1] = trajectory.get_noisy_y()

            for index in range(0, number_of_points_for_msd):
                ea_msd[j_index, index] = np.sum(np.abs((positions[1+index] - positions[0]) ** 2))
                mu_t[j_index, index] = np.sum(np.abs((positions[1+index] - positions[0])))

        ea_msd = np.mean(ea_msd, axis=0)
        mu_t = np.mean(mu_t, axis=0)

        alpha_1 = chi2.ppf(alpha/2, len(trajectories))
        alpha_2 = chi2.ppf(1-(alpha/2), len(trajectories))

        A = (ea_msd-(mu_t**2))*len(trajectories)

        intervals = [
            (A/alpha_1)+(mu_t**2),
            (A/alpha_2)+(mu_t**2)
        ]

        return ea_msd, intervals
        """
        #print("len average ->", np.mean([t.length for t in trajectories]))
        ea_msd = defaultdict(lambda: [])

        delta = np.min(np.diff(trajectories[0].get_time())) if bin_width is None else bin_width

        for trajectory in trajectories:
            positions = np.zeros((trajectory.length,2))
            positions[:,0] = trajectory.get_noisy_x()
            positions[:,1] = trajectory.get_noisy_y()

            for index in range(1, trajectory.length):
                interval = trajectory.get_time()[index] - trajectory.get_time()[0]
                displacement = np.sum(np.abs((positions[index] - positions[0]) ** 2))
                ea_msd[int(interval/delta)].append(displacement)

        intervals = [[], []]

        for i in ea_msd:
            res = bootstrap(ea_msd[i], np.mean, n_resamples=len(trajectories), confidence_level=alpha, method='percentile')
            ea_msd[i] = np.mean(ea_msd[i])
            intervals[0].append(res.confidence_interval.low)
            intervals[1].append(res.confidence_interval.high)

        aux = np.array(sorted(list(zip(list(ea_msd.keys()), list(ea_msd.values()))), key=lambda x: x[0]))
        t_vec, ea_msd = (aux[:,0] * delta) + delta, aux[:,1]

        return t_vec, ea_msd, [np.array(intervals[0]), np.array(intervals[1])]

    def __init__(self, x, y=None, z=None, model_category=None, noise_x=None, noise_y=None, noise_z=None, noisy=False, t=None, exponent=None, exponent_type='anomalous', info={}, **kwargs):

        if exponent_type == "anomalous":
            self.anomalous_exponent = exponent
        elif exponent_type == "hurst":
            self.anomalous_exponent = exponent * 2
        elif exponent_type is None:
            self.anomalous_exponent = None
        else:
            raise Exception(
                f"{exponent_type} exponent type is not available. Use 'anomalous' or 'hurst'.")

        self.model_category = model_category
        
        super().__init__(
            x=x,
            y=y,
            z=z,
            t=t,
            noise_x=noise_x,
            noise_y=noise_y,
            noise_z=noise_z,
            noisy=noisy,
            info=info,
            **kwargs
        )

    """
    Below method only works for bidimension trajectories
    """
    def reconstructed_trajectory(self, delta_t, with_noise=True):
        if with_noise:
            x = self.get_noisy_x()
            y = self.get_noisy_y()
        else:
            x = self.get_x()
            y = self.get_y()

        t = self.get_time()

        new_x = []
        new_y = []
        new_t = []

        for i in range(len(x)):
            t_right = t >= t[i]
            t_left = t <= t[i]+delta_t
            result = np.logical_and(t_right, t_left)

            if np.sum(result) >= 0:
                new_x.append(np.mean(x[result]))
                new_y.append(np.mean(y[result]))
                new_t.append(delta_t*i)

        return Trajectory(
            x = new_x,
            y = new_y,
            t = new_t,
            noisy=True
        )

    def get_anomalous_exponent(self):
        if self.anomalous_exponent is None:
            return "Not available"
        else:
            return self.anomalous_exponent

    def get_model_category(self):
        if self.model_category is None:
            return "Not available"
        else:
            return self.model_category

    @property
    def length(self):
        return len(self.x)

    def get_x(self):
        if self.x is None:
            raise Exception("x was not given")
        return np.copy(np.reshape(self.x, (len(self.x))))

    def get_y(self):
        if self.y is None:
            raise Exception("y was not given")
        return np.copy(np.reshape(self.y, (len(self.y))))
    
    """
    3D Methods are ignored
    def get_z(self):
        if self.z is None:
            raise Exception("y was not given")
        return np.copy(np.reshape(self.z, (len(self.z))))
    """

    @property
    def duration(self):
        return self.get_time()[-1] - self.get_time()[0]

    def get_time(self):
        if self.t is None:
            raise Exception("Time was not given")
        return np.copy(np.copy(np.reshape(self.t, (len(self.t)))))

    def get_noise_x(self):   
        return np.copy(self.noise_x)

    def get_noise_y(self):
        return np.copy(self.noise_y)

    def get_noise_z(self):
        return np.copy(self.noise_z)

    def get_noisy_x(self):   
        if self.noisy:
            return self.get_x()
        
        if self.noise_x is None:
            raise Exception('no x noise was provided')
        else:
            return self.get_x() + np.array(self.noise_x)

    def get_noisy_y(self):
        if self.noisy:
            return self.get_y()
        
        if self.noise_y is None:
            raise Exception('no y noise was provided')
        else:
            return self.get_y() + np.array(self.noise_y)

    """
    3D Methods are ignored
    def get_noisy_z(self):
        if self.noisy:
            return self.get_z()
        
        if self.noise_z is None:
            raise Exception('no z noise was provided')
        else:
            return self.get_z() + np.array(self.noise_z)
    """

    def displacements_on_x(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_x())
        else:
            return np.diff(self.get_x())

    def displacements_on_y(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_y())
        else:
            return np.diff(self.get_y())

    """
    3D Methods are ignored
    def displacements_on_z(self, with_noise=False):
        if with_noise:
            return np.diff(self.get_noisy_z())
        else:
            return np.diff(self.get_z())
    """

    def hurst_exponent(self):
        return self.anomalous_exponent / 2

    def plot(self, with_noise=True):
        if self.model_category is None:

            if self.noisy:
                plt.plot(self.get_noisy_x(), self.get_noisy_y(), marker="X", color='black')
            else:
                plt.plot(self.get_x(), self.get_y(), marker="X", color='black')
                if with_noise:
                    plt.plot(self.get_noisy_x(), self.get_noisy_y(), marker="X", color='red')
            plt.show()
        else:
            self.model_category.plot(self, with_noise=with_noise)

    def plot_andi_2(self, with_noise=True, absolute_d=False, show_break_points=False):
        if self.noisy:
            x, y = self.get_noisy_x(), self.get_noisy_y()
        else:
            if with_noise:
                x, y = self.get_x(), self.get_y()
            if with_noise:
                x, y = self.get_noisy_x(), self.get_noisy_y()

        #fig, ax = plt.subplots(1,3)
        titles = ['State', 'Diffusion Coefficient', 'Anomalous Exponent']
        labels = ['trap', 'confined', 'free', 'directed']
        colors = ['red', 'green', 'blue', 'orange']
        state_to_color = {index:a_color for index, a_color in enumerate(colors)}
        label_to_color = {label:a_color for label, a_color in zip(labels, colors)}

        fig = plt.figure(layout="constrained")
        gs = GridSpec(3, 3, figure=fig)

        ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        ax_regression_alpha = fig.add_subplot(gs[1, :])
        ax_regression_d = fig.add_subplot(gs[2, :])

        """
        state_to_color = {1:confinement_color, 0:non_confinement_color}
        states_as_color = np.vectorize(state_to_color.get)(self.confinement_states(v_th=v_th, window_size=window_size, transition_fix_threshold=transition_fix_threshold))

        for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
            plt.plot([x1, x2], [y1, y2], states_as_color[i], alpha=alpha)
        """

        for i, ax_i in enumerate(ax):
            ax_i.set_title(titles[i])

            if i==0:
                states_as_color = np.vectorize(state_to_color.get)(self.info['state_t'])
                for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
                    ax_i.plot([x1, x2], [y1, y2], states_as_color[i], alpha=1)
                patches = [mpatches.Patch(color=label_to_color[label], label=label.capitalize()) for label in label_to_color]
                ax_i.legend(handles=patches)
            elif i==1:
                norm = plt.Normalize(-12, 1)
                Blues = plt.get_cmap('viridis')
                for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
                    ax_i.plot([x1, x2], [y1, y2], c=Blues(norm(np.log10(self.info['d_t'][i]))), alpha=1)
            elif i==2:
                norm = plt.Normalize(0, 2)
                Blues = plt.get_cmap('viridis')
                for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
                    ax_i.plot([x1, x2], [y1, y2], c=Blues(norm(self.info['alpha_t'][i])), alpha=1)
            x_lim = ax_i.get_xlim()
            y_lim = ax_i.get_ylim()

            x_width = x_lim[1]-x_lim[0]
            y_height = y_lim[1]-y_lim[0]

            if x_width > y_height:
                y_middle = y_lim[0] + (y_height/2)
                ax_i.set_ylim([y_middle-(x_width/2),y_middle+(x_width/2)])
            else:
                x_middle = x_lim[0] + (x_width/2)
                ax_i.set_xlim([x_middle-(y_height/2),x_middle+(y_height/2)])

            ax_i.set_aspect('equal', adjustable='box')

        ax_regression_d.set_ylabel(r'$D_{i}$')
        ax_regression_d.set_xlabel(r'$i$')

        if not absolute_d:
            ax_regression_d.plot(np.log10(self.info['d_t']))
            if show_break_points:
                break_points = rpt.Pelt(model="l1").fit(np.log10(self.info['d_t'])).predict(pen=1)
                for bkp in break_points:
                    ax_regression_d.axvline(bkp, color='black', linestyle='-', linewidth=2)
            ax_regression_d.set_ylim([-12,6])
        else:
            ax_regression_d.plot(self.info['d_t'])
            if show_break_points:
                break_points = rpt.Pelt(model="l1").fit(self.info['d_t']).predict(pen=1)
                for bkp in break_points:
                    ax_regression_d.axvline(bkp, color='black', linestyle='-', linewidth=2)

        ax_regression_alpha.plot(self.info['alpha_t'])

        if show_break_points:
            break_points = rpt.Pelt(model="l1").fit(self.info['alpha_t']).predict(pen=1)
            for bkp in break_points:
                ax_regression_alpha.axvline(bkp, color='black', linestyle='-', linewidth=2)

        ax_regression_alpha.set_ylabel(r'$\alpha_{i}$')
        #ax_regression_alpha.set_xlabel(r'$i$')
        ax_regression_alpha.set_ylim([0,2])
        #manager = plt.get_current_fig_manager()
        #manager.full_screen_toggle()
        plt.show()

    def animate_plot(self, roi_size=None, save_animation=False, title='animation'):
        fig, ax = plt.subplots()
        line = ax.plot(self.get_noisy_x()[0], self.get_noisy_y()[0])[0]

        if roi_size is None:
            ax.set(xlim=[np.min(self.get_noisy_x()), np.max(self.get_noisy_x())], ylim=[np.min(self.get_noisy_y()), np.max(self.get_noisy_y())], xlabel='X', ylabel='Y')
        else:
            xlim = [np.min(self.get_noisy_x()), np.max(self.get_noisy_x())]
            ylim = [np.min(self.get_noisy_y()), np.max(self.get_noisy_y())]
            x_difference = xlim[1]-xlim[0]
            y_difference = ylim[1]-ylim[0]
            x_offset = (roi_size - x_difference)/2
            y_offset = (roi_size - y_difference)/2
            xlim = [xlim[0]-x_offset, xlim[1]+x_offset]
            ylim = [ylim[0]-y_offset, ylim[1]+y_offset]
            ax.set(xlim=xlim, ylim=ylim, xlabel='X', ylabel='Y')
        def update(frame):
            # for each frame, update the data stored on each artist.
            x_f = self.get_noisy_x()[:frame]
            y_f = self.get_noisy_y()[:frame]

            if self.t is not None:
                time = (self.get_time() - self.get_time()[0])[frame]
                time = np.round(time, 6)
                ax.set_title(f'{time}s')
            #voronoi_plot_2d(Voronoi(self.model_category.voronoi_centroids), show_points=False, show_vertices=False, line_colors='grey', ax=ax)
            # update the scatter plot:
            #data = np.stack([x, y]).T
            # update the line plot:
            line.set_xdata(x_f[:frame])
            line.set_ydata(y_f[:frame])
            plt.tight_layout()
            return (line)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.length, interval=1)

        if not save_animation:
            plt.show()
        else:
            ani.save(f'DELETE.gif', writer=animation.PillowWriter(fps=30), dpi=300)
            clip = mp.VideoFileClip(f'DELETE.gif')
            clip.write_videofile(f'./animations_plus/{title}.mp4')

    def plot_confinement_states(
        self,
        v_th=11,
        window_size=3,
        transition_fix_threshold=9,
        non_confinement_color='black',
        confinement_color='green',
        show=True,
        alpha=1,
        plot_confinement_convex_hull=False,
        color_confinement_convex_hull='grey',
        alpha_confinement_convex_hull=0.5
    ):
        x = self.get_noisy_x().tolist()
        y = self.get_noisy_y().tolist()

        state_to_color = {1:confinement_color, 0:non_confinement_color}
        states_as_color = np.vectorize(state_to_color.get)(self.confinement_states(v_th=v_th, window_size=window_size, transition_fix_threshold=transition_fix_threshold))

        for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
            plt.plot([x1, x2], [y1, y2], states_as_color[i], alpha=alpha)

        confinement_sub_trajectories = self.sub_trajectories_trajectories_from_confinement_states(v_th=v_th, window_size=window_size, transition_fix_threshold=transition_fix_threshold)[1]

        if plot_confinement_convex_hull:
            for trajectory in confinement_sub_trajectories:
                points = np.zeros((trajectory.length, 2))
                points[:,0] = trajectory.get_noisy_x()
                points[:,1] = trajectory.get_noisy_y()
                hull = ConvexHull(points)

                plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color_confinement_convex_hull, alpha=alpha_confinement_convex_hull)
        
        if show:
            plt.show()

    def __str__(self):
        anomalous_exponent_string = "%.2f" % self.anomalous_exponent if self.anomalous_exponent is not None else None
        return f"Model: {self.model_category}, Anomalous Exponent: {anomalous_exponent_string}, Trajectory Length: {self.length}"

    @property
    def normalized_ratio(self):
        r = 0
        delta_r = []

        # Extract coordinate values from track
        data = np.zeros(shape=(2, self.length))
        data[0, :] = self.get_noisy_x()
        data[1, :] = self.get_noisy_y()

        for j in range(self.length):
            r = r + np.linalg.norm([data[0, j] - np.mean(data[0, :]), data[1, j] - np.mean(data[1, :])]) ** 2

        for j in range(self.length - 1):
            delta_r.append(np.linalg.norm([data[0, j + 1] - data[0, j], data[1, j + 1] - data[1, j]]))

        rad_gir = np.sqrt((1 / self.length) * r)
        mean_delta_r = np.mean(delta_r)
        criteria = (rad_gir / mean_delta_r) * np.sqrt(np.pi/2)
        return float(criteria)

    def is_immobile(self, threshold):
        return self.normalized_ratio <= threshold

    def sub_trajectories_trajectories_from_confinement_states(self, v_th=11, window_size=3, transition_fix_threshold=9, use_info=False):
        confinement_states = self.confinement_states(return_intervals=False, v_th=v_th, transition_fix_threshold=transition_fix_threshold, window_size=window_size) if not use_info else self.info['analysis']['confinement-states']

        trajectories = {
            0: [],
            1: []
        }

        def divide_list(a_list):
            sublists = []
            current_sublist = []
            
            for element in a_list:
                if len(current_sublist) == 0 or element == current_sublist[0]:
                    current_sublist.append(element)
                else:
                    sublists.append(current_sublist)
                    current_sublist = [element]

            if len(current_sublist) > 0:
                sublists.append(current_sublist)

            return sublists

        index = 0

        for sublist in divide_list(confinement_states):
            trajectories[sublist[0]].append(self.build_noisy_subtrajectory_from_range(index, index+len(sublist)))
            index += len(sublist)
        
        return trajectories

    def build_noisy_subtrajectory_from_range(self, initial_index, final_index, noisy=True):
        new_trajectory = Trajectory(
                    x = self.get_noisy_x()[initial_index:final_index],
                    y = self.get_noisy_y()[initial_index:final_index],
                    t = self.get_time()[initial_index:final_index],
                    noisy=noisy
                )
        
        BTX_NOMENCLATURE = 'BTX680R'
        CHOL_NOMENCLATURE = 'fPEG-Chol'

        if 'dcr' in self.info:
            new_trajectory.info['dcr'] = self.info['dcr'][initial_index:final_index]
        if 'intensity' in self.info:
            new_trajectory.info['intensity'] = self.info['intensity'][initial_index:final_index]
        if 'dataset' in self.info:
            new_trajectory.info['dataset'] = self.info['dataset']
        if 'roi' in self.info:
            new_trajectory.info['roi'] = self.info['roi']
        if 'file' in self.info:
            new_trajectory.info['file'] = self.info['file']
        if 'classified_experimental_condition' in self.info:
            new_trajectory.info['classified_experimental_condition'] = self.info['classified_experimental_condition']
        if f'{BTX_NOMENCLATURE}_single_intersections' in self.info:
            new_trajectory.info[f'{BTX_NOMENCLATURE}_single_intersections'] = self.info[f'{BTX_NOMENCLATURE}_single_intersections'][initial_index:final_index]
        if f'{CHOL_NOMENCLATURE}_single_intersections' in self.info:
            new_trajectory.info[f'{CHOL_NOMENCLATURE}_single_intersections'] = self.info[f'{CHOL_NOMENCLATURE}_single_intersections'][initial_index:final_index]

        return new_trajectory

    def confinement_states(self,v_th=11, window_size=3, transition_fix_threshold=9, return_intervals=False):
        """
        This method is the Array-Oriented Python implementation of the algorithm proposed in the referenced
        paper to identify periods of transient confinement within individual trajectories.

        Sikora, G., Wyłomańska, A., Gajda, J., Solé, L., Akin, E. J., Tamkun, M. M., & Krapf, D. (2017).

        Elucidating distinct ion channel populations on the surface of hippocampal neurons via single-particle
        tracking recurrence analysis. Physical review. E, 96(6-1), 062404.
        https://doi.org/10.1103/PhysRevE.96.062404
        """
        if self.length == 1:
            if return_intervals:
                return [0], []
            else:
                return [0]

        C = self.length-1

        X = np.zeros((self.length,2))
        X[:,0] = self.get_noisy_x()
        X[:,1] = self.get_noisy_y()

        M = (X[:-1] + X[1:])/2
        R = np.linalg.norm(X[:-1] - X[1:], axis=1)/2

        S = scipy.sparse.lil_matrix(np.zeros((self.length, C)))

        for position_index in range(self.length):
            distances = scipy.spatial.distance_matrix(np.array([X[position_index]]), M)
            S[position_index, :] = (distances < R).astype(int)

        V = np.array(np.sum(S, axis=0))[0]
        V_convolved = np.convolve(V, np.ones(window_size))
        V = np.repeat(V_convolved[window_size-1::window_size], window_size)[:C]
        V = (V > v_th).astype(int)

        states = np.zeros(self.length)

        for position_index in range(self.length):
            states[position_index] = np.sum(S[position_index, :] * V)

        states = (states > 0).astype(int)

        #Spurious transitions are eliminated
        for window_index in range(0,len(states), transition_fix_threshold):
            states[window_index:window_index+transition_fix_threshold] = np.argmax(np.bincount(states[window_index:window_index+transition_fix_threshold]))

        if return_intervals:
            indices = np.nonzero(states[1:] != states[:-1])[0] + 1
            intervals = np.split(self.get_time(), indices)
            intervals = intervals[0::2] if states[0] else intervals[1::2]
            intervals = [interval for interval in intervals if interval[-1] - interval[0] != 0]

            return states, intervals
        else:
            return states

    def calculate_msd_curve(self, with_noise=True, bin_width=None):
        """
        Code Obtained from https://github.com/Eggeling-Lab-Microscope-Software/TRAIT2D/blob/b51498b730140ffac5c0abfc5494ebfca25b445e/trait2d/analysis/__init__.py#L1061
        """
        if with_noise:
            x = self.get_noisy_x()
            y = self.get_noisy_y()
        else:
            x = self.get_x()
            y = self.get_y()

        N = len(x)
        assert N-3 > 0
        col_Array  = np.zeros(N-3)
        col_t_Array  = np.zeros(N-3)
        data_tmp = np.column_stack((x, y))
        data_t_tmp = self.get_time()

        msd_dict = defaultdict(lambda: [])

        delta = np.min(np.diff(self.get_time())) if bin_width is None else bin_width

        for i in range(1,N-2):
            calc_tmp = np.sum(np.abs((data_tmp[1+i:N,:] - data_tmp[1:N - i,:]) ** 2), axis=1)
            calc_t_tmp = data_t_tmp[1+i:N] - data_t_tmp[1:N - i]

            for interval, square_displacement in zip(calc_t_tmp, calc_tmp):
                msd_dict[int(interval/delta)].append(square_displacement)

            col_Array[i-1] = np.mean(calc_tmp)
            col_t_Array[i-1] = i * delta

        for i in msd_dict:
            msd_dict[i] = np.mean(msd_dict[i])

        aux = np.array(sorted(list(zip(list(msd_dict.keys()), list(msd_dict.values()))), key=lambda x: x[0]))
        t_vec, msd = (aux[:,0] * delta) + delta, aux[:,1]

        assert len(t_vec) == len(msd)

        return t_vec, msd

    def temporal_average_mean_squared_displacement(self, non_linear=True, log_log_fit_limit=50, with_noise=True, bin_width=None):
        def real_func(t, betha, k):
            return k * (t ** betha)

        def linear_func(t, betha, k):
            return np.log(k) + (np.log(t) * betha)

        t_vec, msd = self.calculate_msd_curve(with_noise=with_noise, bin_width=bin_width)

        msd_fit = msd[0:log_log_fit_limit]
        t_vec_fit = t_vec[0:log_log_fit_limit]
        assert len(t_vec_fit) == log_log_fit_limit
        assert len(msd_fit) == log_log_fit_limit

        popt, _ = curve_fit(linear_func, t_vec_fit, np.log(msd_fit), bounds=((0, 0), (2, np.inf)), maxfev=2000)
        goodness_of_fit = r2_score(np.log(msd_fit), linear_func(t_vec_fit, popt[0], popt[1]))

        #plt.title(f"betha={np.round(popt[0], 2)}, k={popt[1]}")
        #plt.plot(t_vec_fit, t_vec_fit * popt[1])
        #plt.plot(t_vec_fit, msd_fit)
        #plt.show()
        return t_vec, msd, popt[0], popt[1], goodness_of_fit

    def short_range_diffusion_coefficient_msd(self, with_noise=True, bin_width=None):
        def linear_func(t, d, sigma):
            return (4 * t * d) + (sigma**2)

        if with_noise:
            x = self.get_noisy_x()
            y = self.get_noisy_y()
        else:
            x = self.get_x()
            y = self.get_y()

        t_vec, msd = self.calculate_msd_curve(with_noise=with_noise, bin_width=bin_width)

        msd_fit = msd[1:4]
        t_vec_fit = t_vec[1:4]
        assert len(msd_fit) == 3
        assert len(t_vec_fit) == 3
        popt, _ = curve_fit(linear_func, t_vec_fit, msd_fit, bounds=((0, 0), (np.inf, np.inf)), maxfev=2000)
        goodness_of_fit = r2_score(msd_fit, linear_func(t_vec_fit, popt[0], popt[1]))

        return t_vec, msd, popt[0], popt[1], goodness_of_fit

    def turning_angles(self,steps_lag=1, normalized=False):
        return turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            steps_lag=steps_lag,
            normalized=normalized
        )

    def directional_correlation(self, steps_lag=1, window_size=9):
        return directional_correlation(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            steps_lag=steps_lag,
            window_size=window_size
        )

    def directional_correlation_segmentation(self, steps_lag=1, window_size=9, pen=1, jump=1, min_size=3, return_break_points=False):
        return directional_correlation_segmentation(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            steps_lag=steps_lag,
            window_size=window_size,
            pen=pen,
            jump=jump,
            min_size=min_size,
            return_break_points=return_break_points
        )

    def mean_turning_angle(self):
        """
        This is meanDP in

        Deep learning assisted single particle tracking for
        automated correlation between diffusion and
        function
        """
        normalized_angles = turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            normalized=True,
            steps_lag=1
        )
        return np.nanmean(normalized_angles)

    def correlated_turning_angle(self):
        """
        This is corrDP in

        Deep learning assisted single particle tracking for
        automated correlation between diffusion and
        function
        """
        normalized_angles = turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            normalized=True,
            steps_lag=1
        )
        return np.nanmean(np.sign(normalized_angles[1:])==np.sign(normalized_angles[:-1]))

    def directional_persistance(self):
        """
        This is AvgSignDp in

        Deep learning assisted single particle tracking for
        automated correlation between diffusion and
        function
        """
        normalized_angles = turning_angles(
            self.length,
            self.get_noisy_x(),
            self.get_noisy_y(),
            normalized=True,
            steps_lag=1
        )
        return np.nanmean(np.sign(normalized_angles[1:])>0)
