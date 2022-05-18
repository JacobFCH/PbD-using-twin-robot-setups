import numpy as np
import matplotlib.pyplot as plt


class TrajectoryGenerator:
    def __init__(self, scale, z):
        self.scale = scale
        self.z = z
        self.testTrajectory = self.computeTrajectory()

    def computeTrajectory(self):
        t = 0
        test_trajectory = []
        while t < 6.284:
            x = np.cos(t)
            y = np.sin(2 * t) / 2
            test_trajectory.append([x, y, self.z/self.scale, 1])
            t += 0.002

        T = np.array([[0, 1, 0, -0.5],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        test_trajectory = np.asarray(test_trajectory)

        test_trajectory[:, 0:3] = test_trajectory[:, 0:3] * self.scale

        new_trajectory = []
        for point in test_trajectory:
            translated_point = T @ point
            new_trajectory.append([translated_point[0] / translated_point[3], translated_point[1] / translated_point[3],
                                   translated_point[2] / translated_point[3], 3.14, 0, 0])

        new_trajectory = np.asarray(new_trajectory)

        return new_trajectory

    def dtw(self, s, t):
        n, m = len(s), len(t)
        dtw_matrix = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s[i - 1] - t[j - 1])
                # take last min from a square box
                last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix[n,m]

    def plotTrajectory(self):
        plt.plot(self.testTrajectory[:, 0], self.testTrajectory[:, 1])
        plt.show()
