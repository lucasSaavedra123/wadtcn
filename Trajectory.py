import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

        state_to_color = {1:'red', 0:'black'}
        states_as_color = np.vectorize(state_to_color.get)(self.confinement_states(v_th=v_th, window_size=window_size))

        for i,(x1, x2, y1,y2) in enumerate(zip(x, x[1:], y, y[1:])):
            plt.plot([x1, x2], [y1, y2], states_as_color[i], marker='X')  

        if show:
            plt.show()

    def __str__(self):
        anomalous_exponent_string = "%.2f" % self.anomalous_exponent if self.anomalous_exponent is not None else None
        return f"Model: {self.model_category}, Anomalous Exponent: {anomalous_exponent_string}, Trajectory Length: {self.length}"

    def is_immobile(self, threshold):
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
        
        return criteria <= threshold

    def confinement_states(self,v_th=11,window_size=3):
        """
        This method is the Object-Oriented Python implementation of the algorithm proposed in the referenced 
        paper to identify periods of transient confinement within individual trajectories.

        Sikora, G., Wyłomańska, A., Gajda, J., Solé, L., Akin, E. J., Tamkun, M. M., & Krapf, D. (2017).

        Elucidating distinct ion channel populations on the surface of hippocampal neurons via single-particle
        tracking recurrence analysis. Physical review. E, 96(6-1), 062404.
        https://doi.org/10.1103/PhysRevE.96.062404
        """

        class Circle:
            def __init__(self, point_i, point_j):
                self.point_i = point_i
                self.point_j = point_j

                midpoint_x = (self.point_i[0] + self.point_j[0])/2
                midpoint_y = (self.point_i[1] + self.point_j[1])/2
                self.midpoint = [midpoint_x, midpoint_y]

                self.diameter = math.dist(self.point_i, self.point_j)

                self._count_cache = None
                self._index_points_inside_area_cache = None

            def count_number_of_times_the_walker_position_lies_within_circle(self, points):
                self._count_cache = 0
                self._index_points_inside_area_cache = []
                for index, point in enumerate(points):
                    if math.dist(point, self.midpoint) < (self.diameter/2):
                        self._count_cache += 1
                        self._index_points_inside_area_cache.append(index)

            @property
            def count(self):
                return self._count_cache

            @property
            def index_points_inside_area(self):
                return self._index_points_inside_area_cache

        x = self.get_noisy_x().tolist()
        y = self.get_noisy_y().tolist()

        points = list(zip(x,y))

        circles = []

        for i in range(1,self.length):
            new_circle = Circle(points[i-1], points[i])
            new_circle.count_number_of_times_the_walker_position_lies_within_circle(points)
            circles.append(new_circle)

        states = np.zeros(self.length)

        for sub_circles in [circles[i:i+window_size] for i in range(0,len(circles),window_size)]:
            circles_windows_count = sum([sub_circle.count for sub_circle in sub_circles])

            if circles_windows_count > v_th:
                for sub_circle in sub_circles:
                    states[sub_circle.index_points_inside_area] = 1

        return states.tolist()

    def mean_squared_displacement(self, non_linear=True):
        """
        Code Obtained from https://github.com/hectorbm/DL_anomalous_diffusion/blob/ab13739cb8fdb947dd1ebc9a8f537668eb26266a/Tools/analysis_tools.py#L36C67-L36C67
        """
        def linear_func(x, beta, d):
            return d * (x ** 1)

        x = self.get_noisy_x()
        y = self.get_noisy_y()
        time_length = (self.get_time()[-1] - self.get_time()[0])
        data = np.sqrt(x ** 2 + y ** 2)
        n_data = np.size(data)
        number_of_delta_t = np.int((n_data - 1))
        t_vec = np.arange(1, np.int(number_of_delta_t))

        msd = np.zeros([len(t_vec), 1])
        for dt, ind in zip(t_vec, range(len(t_vec))):
            squared_displacement = (data[1 + dt:] - data[:-1 - dt]) ** 2
            msd[ind] = np.mean(squared_displacement, axis=0)

        msd = np.array(msd)

        t_vec = np.linspace(0.0001, time_length, len(x) - 2)
        msd = np.array(msd).ravel()
        if non_linear:
            a, b = curve_fit(linear_func, t_vec, msd, bounds=((0, 0), (2, np.inf)), maxfev=2000)
        else:
            a, b = curve_fit(linear_func, t_vec, msd, bounds=((0, 0, -np.inf), (2, np.inf, np.inf)), maxfev=2000)

        return t_vec, msd, a
