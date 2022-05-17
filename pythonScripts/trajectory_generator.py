import numpy as np
import matplotlib.pyplot as plt

t = 0
test_trajectory = []
while t < 6.284:
    x = np.cos(t)
    y = np.sin(2 * t) / 2
    test_trajectory.append([x, y])
    t += 0.002

#np.array([[],
#          [],
#          [],
#          [0, 0, 0, 1]])

test_trajectory = np.asarray(test_trajectory)

test_trajectory = test_trajectory * 0.25

plt.plot(test_trajectory[:,0], test_trajectory[:,1])
plt.show()