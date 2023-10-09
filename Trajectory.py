import math

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import ruptures as rpt
from mongoengine import Document, FloatField, ListField, DictField, BooleanField

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
    if length/steps_lag <= 2:
        return []

    X = np.zeros((length,2))
    X[:,0] = x
    X[:,1] = y

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
    def ensamble_average_mean_square_displacement(cls, trajectories, number_of_points_for_msd=50, alpha=0.95):
        trajectories = [trajectory for trajectory in trajectories if trajectory.length > number_of_points_for_msd + 1]

        ea_msd = np.zeros((len(trajectories), number_of_points_for_msd))
        mu_t = np.zeros((len(trajectories), number_of_points_for_msd))

        for j_index, trajectory in enumerate(trajectories):
            positions = np.zeros((trajectory.length,2))
            positions[:,0] = trajectory.get_noisy_x()
            positions[:,1] = trajectory.get_noisy_y()

            for index in range(0, number_of_points_for_msd):
                ea_msd[j_index, index] = np.linalg.norm(positions[index+1]-positions[0]) ** 2
                mu_t[j_index, index] = np.linalg.norm(positions[index+1]-positions[0])

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
    def reconstructed_trajectory(self, delta_t):
        x = self.get_noisy_x()
        y = self.get_noisy_y()
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

    def plot(self, axis='xy'):
        plt.title(self)
        if axis == 'x':
            plt.plot(self.get_x(), marker="X")
            plt.plot(self.get_noisy_x(), marker="X")
        elif axis == 'y':
            plt.plot(self.get_y(), marker="X")
            plt.plot(self.get_noisy_y(), marker="X")        
        elif axis == 'xy':
            plt.plot(self.get_x(), self.get_y(), marker="X")
            plt.plot(self.get_noisy_x(), self.get_noisy_y(), marker="X")

        plt.show()

    def plot_confinement_states(self, v_th=11, window_size=3, show=True):
        x = self.get_noisy_x().tolist()
        y = self.get_noisy_y().tolist()

        state_to_color = {1:'green', 0:'black'}
        states_as_color = np.vectorize(state_to_color.get)(self.confinement_states(v_th=v_th, window_size=window_size))

        for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
            plt.plot([x1, x2], [y1, y2], states_as_color[i])

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

    def sub_trajectories_trajectories_from_confinement_states(self, v_th=11, window_size=3):
        confinement_states = self.confinement_states(return_intervals=False, v_th=v_th, window_size=window_size)

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

    def build_noisy_subtrajectory_from_range(self, initial_index, final_index):
        return Trajectory(
                    x = self.get_noisy_x()[initial_index:final_index],
                    y = self.get_noisy_y()[initial_index:final_index],
                    t = self.get_time()[initial_index:final_index],
                    noisy=True,
                    info=self.info,
                    exponent=self.anomalous_exponent
                )

    def confinement_states(self,v_th=11,window_size=3, return_intervals=False):
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

        if return_intervals:
            indices = np.nonzero(states[1:] != states[:-1])[0] + 1
            intervals = np.split(self.get_time(), indices)
            intervals = intervals[0::2] if states[0] else intervals[1::2]
            intervals = [interval for interval in intervals if interval[-1] - interval[0] != 0]

            return states, intervals
        else:
            return states

    def temporal_average_mean_squared_displacement(self, non_linear=True, log_log_fit_limit=50):
        """
        Code Obtained from https://github.com/hectorbm/DL_anomalous_diffusion/blob/ab13739cb8fdb947dd1ebc9a8f537668eb26266a/Tools/analysis_tools.py#L36C67-L36C67
        """
        def real_func(t, betha, k):
            return k * (t ** betha)

        def linear_func(t, betha, k):
            return np.log(k) + (np.log(t) * betha)

        x = self.get_noisy_x()
        y = self.get_noisy_y()
        time_length = self.duration
        data = np.sqrt(x ** 2 + y ** 2)
        number_of_delta_t = self.length - 1
        t_vec = np.arange(1, number_of_delta_t)

        msd = np.zeros(len(t_vec))
        for index, dt in enumerate(t_vec):
            squared_displacement = (data[1 + dt:] - data[:-1 - dt]) ** 2
            msd[index] = np.mean(squared_displacement, axis=0)

        t_vec = np.linspace(self.get_time()[1] - self.get_time()[0], time_length, self.length - 2)
        #t_vec = self.get_time() - self.get_time()[0]
        #t_vec = np.linspace(0, self.length-2,1) * 

        msd_fit = msd[0:log_log_fit_limit]
        t_vec_fit = t_vec[0:log_log_fit_limit]

        popt, _ = curve_fit(linear_func, t_vec_fit, np.log(msd_fit), bounds=((0, 0), (2, np.inf)), maxfev=2000)
        
        """
        if non_linear:
            popt, _ = curve_fit(real_func, t_vec_fit, msd_fit, bounds=((0, 0), (2, np.inf)), maxfev=2000)
            #popt2, _ = curve_fit(linear_func, t_vec_fit, np.log(msd_fit), bounds=((0, 0), (2, np.inf)), maxfev=2000)
        else:
            popt, _ = curve_fit(real_func, t_vec_fit, msd_fit, bounds=((0, 0, -np.inf), (2, np.inf, np.inf)), maxfev=2000)
        """

        goodness_of_fit = r2_score(np.log(msd_fit), linear_func(t_vec_fit, popt[0], popt[1]))

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
