import numpy as np
import matplotlib.pyplot as plt


class TrajectoryGenerator:
    def __init__(self, scale):
        self.scale = scale
        self.testTrajectory = self.computeTrajectory()

    def computeTrajectory(self):
        t = 0
        test_trajectory = []
        while t < 6.284:
            x = np.cos(t)
            y = np.sin(2 * t) / 2
            test_trajectory.append([x, y, 0.30/self.scale, 1])
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

    def plotTrajectory(self):
        plt.plot(self.testTrajectory[:, 0], self.testTrajectory[:, 1])
        plt.show()