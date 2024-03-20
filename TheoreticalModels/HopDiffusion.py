import numpy as np

from TheoreticalModels.Model import Model
from TheoreticalModels.simulation_utils import add_noise_and_offset, simulate_track_time

from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi, voronoi_plot_2d
from CONSTANTS import EXPERIMENT_WIDTH, EXPERIMENT_HEIGHT
import matplotlib.pyplot as plt

class HopDiffusion(Model):
    STRING_LABEL = 'hd'
    D_RANGE = [0.001, 1] #um2/s
    P_HOP_RANGE = [0.1, 0.5]

    @classmethod
    def create_random_instance(cls):
        p_hop = np.random.uniform(low=cls.P_HOP_RANGE[0], high=cls.P_HOP_RANGE[1])
        d = np.random.choice(np.logspace(np.log10(cls.D_RANGE[0]), np.log10(cls.D_RANGE[1]), 1000))
        return cls(d, p_hop)

    def __init__(self, d, p_hop):
        assert d > 0
        assert 0 < p_hop < 1
        self.d = d * 1000000 #um2/s -> nm2/s
        self.p_hop = p_hop
        self.roi = (EXPERIMENT_HEIGHT+EXPERIMENT_WIDTH)/2
        self.l = np.random.uniform(10,1000)#nm
        self.__voronoi_centroids = np.random.uniform(0, self.roi, size=(int(self.roi/self.l)**2, 2))

    def __extract_polygons_from_voronoi(self):
        polygons = []
        voronoi_diagram = Voronoi(self.__voronoi_centroids)
        for region in voronoi_diagram.regions:
            if not -1 in region and region != []:
                polygon = Polygon([voronoi_diagram.vertices[i] for i in region])
                polygons.append(polygon)
            else:
                polygons.append(None)
        return polygons

    def __get_voronoi_centroids(self):
        centroids = []
        for polygon in self.__extract_polygons_from_voronoi():
            if polygon is not None:
                centroid = polygon.centroid
                centroids.append([centroid.x, centroid.y])
            else:
                centroids.append([np.inf, np.inf])
        return np.array(centroids)

    def __get_region_of_position(self,x,y):
        xd = (x - self.__voronoi_centroids[:,0])**2
        yd = (y - self.__voronoi_centroids[:,1])**2
        distances = np.sqrt(xd + yd)
        min_index = np.argmin(distances)
        return min_index

    def custom_simulate_rawly(self, trajectory_length, trajectory_time):
        cells_centroids = self.__get_voronoi_centroids()
        
        initial_position = cells_centroids[np.random.randint(cells_centroids.shape[0])]
        while np.inf in initial_position:
            initial_position = cells_centroids[np.random.randint(cells_centroids.shape[0])]

        x, y = [initial_position[0]], [initial_position[1]]
        current_region = self.__get_region_of_position(x[0],y[0])

        switching = False

        while len(x) != trajectory_length:
            x_next_position = x[-1] + np.random.normal(loc=0, scale=1) * np.sqrt(2 * self.d * (trajectory_time/trajectory_length))
            y_next_position = y[-1] + np.random.normal(loc=0, scale=1) * np.sqrt(2 * self.d * (trajectory_time/trajectory_length))

            next_region = self.__get_region_of_position(x_next_position, y_next_position)

            if current_region == next_region:
                x.append(x_next_position)
                y.append(y_next_position)
            else:
                switching = True
                if np.random.choice([True,False], p=[self.p_hop, 1-self.p_hop]):
                    x.append(x_next_position)
                    y.append(y_next_position)
                    current_region = next_region

        x, x_noisy, y, y_noisy = add_noise_and_offset(trajectory_length, np.array(x), np.array(y), disable_offset=True)
        t = simulate_track_time(trajectory_length, trajectory_time)

        return {
            'x': x,
            'y': y,
            't': t,
            'x_noisy': x_noisy,
            'y_noisy': y_noisy,
            'exponent_type': None,
            'exponent': None,
            'info': {
                'switching': switching,
                'p_hop': self.p_hop,
                'd': self.d,
            }
        }

    def plot(self, trajectory, with_noise=False):
        fig = voronoi_plot_2d(Voronoi(self.__voronoi_centroids), show_points=False, show_vertices=False, line_colors='grey')

        plt.suptitle(r"$P_{Hop}="+str(np.round(self.p_hop, 2))+r"$, $D="+str(np.round(self.d/1000000, 3))+r"\mu m^{2}/s$")
        plt.plot(trajectory.get_x(), trajectory.get_y(), marker="X", color='black')
        if with_noise:
            plt.plot(trajectory.get_noisy_x(), trajectory.get_noisy_y(), marker="X", color='red')

        plt.xlim([np.min(trajectory.get_x()) * 0.95, np.max(trajectory.get_x()) * 1.05])
        plt.ylim([np.min(trajectory.get_y()) * 0.95, np.max(trajectory.get_y()) * 1.05])

        plt.show()
