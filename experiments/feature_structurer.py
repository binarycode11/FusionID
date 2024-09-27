from typing import Any, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from interfaces.pipeline import IGlobalFeatureStructurer

class DelaunayGraph(IGlobalFeatureStructurer):
    @staticmethod
    def distancePoint(p1, p2):
        return np.linalg.norm(p1 - p2)#math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def buildConnection(self, tri):
        mapConnection = []
        for simplex in tri.simplices:
            mapConnection.extend([
                [simplex[0], simplex[1]],
                [simplex[1], simplex[2]],
                [simplex[2], simplex[0]]
            ])
        return mapConnection

    def buildMapGraph(self, mapConex, featuresByPoints):
        size = len(featuresByPoints)
        sample = np.matrix(np.ones((size, size)) * np.inf)
        for i in range(size):
            sample[i, i] = 0
        for conexao in mapConex:
            distance = self.distancePoint(featuresByPoints[conexao[0]], featuresByPoints[conexao[1]])
            sample[conexao[0], conexao[1]] = distance
        return sample

    @staticmethod
    def plot_delaunay(points, tri, img):
        plt.imshow(img)
        plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), color='#0000FF')
        plt.plot(points[:, 0], points[:, 1], 'o', color='#6495ED')
        dist = 3
        for i, point in enumerate(points):
            plt.text(point[0] + dist, point[1] + dist, f' {i}', color='#6495ED', fontsize=12)
        plt.show()

    def __call__(self, points: np.ndarray, featuresByPoints: np.ndarray) -> Tuple[np.ndarray, Any]:
        tri = Delaunay(points)
        mapConnection = self.buildConnection(tri)
        graph = self.buildMapGraph(mapConnection, featuresByPoints)
        return graph, tri
