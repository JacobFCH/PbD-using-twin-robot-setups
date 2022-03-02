from numbers import Real
import numpy as np
import roboticstoolbox as rbt
from spatialmath import *
from scipy.spatial.transform.rotation import Rotation as R
import time

class ikSolver():
    def __init__(self):
        self.a = [0, -0.425, -0.39225, 0, 0, 0]
        self.d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        self.alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

    def solveIK(self, T06):
        qs = np.asarray([])
        theta = np.zeros([8,6])

        # ---------- Theta 1 ----------
        P05 = (T06 @ np.array([0,0,-self.d[5], 1]))[0:3]
        phi1 = np.arctan2(P05[1],P05[0])
        phi2 = np.array([np.arccos(self.d[3]/np.linalg.norm(P05[0:1])), -np.arccos(self.d[3]/np.linalg.norm(P05[0:1]))])

        for i in range(4):
            theta[i,0] = phi1 + phi2[0] + np.pi/2
            theta[i+4,0] = phi1 + phi2[1] + np.pi/2

        for i in range(8):
            if theta[i,0] <= np.pi:
                theta[i,0] += 2*np.pi
            if theta[i,0] > np.pi:
                theta[i,0] -= 2*np.pi
        
        # ---------- Theta 5 ----------
        P06 = T06[0:3,3]
        for i in range(8):
            theta[i,4] = np.arccos((P06[0]*np.sin(theta[i,0])-P06[1]*np.cos(theta[i,0])-self.d[3])/self.d[5])
            if np.isin(i, [2,3,6,7]):
                theta[i,4] = -theta[i,4]
        print(theta)
        return qs


frame = np.array([-0.42, 0.0235, 0.45, 0, 0, 0 ])
#configuration = np.array([-0.3185, 0.2370,  4.2150, 0.2534, 1.5708, 1.8867])

T = np.eye(4)
rot = R.from_euler('xyz', frame[3:6])
T[0:3,0:3] = rot.as_matrix()
T[0:3,3] = frame[0:3]

ik = ikSolver()
ik.solveIK(T)