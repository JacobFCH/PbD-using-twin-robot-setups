from cmath import inf
import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform.rotation import Rotation as R

# Inverse kinematic solver for a UR robot, currently using DH parameters of a UR5e
# Based on Kinematics of a UR5, by Rasmus Skovgaard Andersen
# http://rasmusan.blog.aau.dk/files/ur5_kinematics.pdf

class ikSolver():
    def __init__(self):
        #self.d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]

        self.a = np.array([0, -0.425, -0.39225, 0, 0, 0])
        self.d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
        self.alpha = np.array([0, np.deg2rad(90), 0, 0, np.deg2rad(90), np.deg2rad(-90)]) # This is moved one sport over see paper

    def DHLink(self, alpha, a, d, angle):
        T = np.array([[np.cos(angle),                 -np.sin(angle),                0,              a],
                     [np.sin(angle) * np.cos(alpha), np.cos(angle) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                     [np.sin(angle) * np.sin(alpha), np.cos(angle) * np.sin(alpha), np.cos(alpha),  np.cos(alpha)*d],
                     [0,                             0,                             0,              1]])
        return T

    def nearestQ(self, qs, last_q):
        weights = np.array([6,5,4,3,2,1])
        best_q = np.zeros(6)
        bestConfDist = np.inf
        for q in qs:
            confDist = np.sum(((q - last_q) * weights)**2)
            if confDist < bestConfDist:
                bestConfDist = confDist
                best_q = q
        return np.asarray(best_q)

    def solveIK(self, T06, last_q):
        theta = np.zeros([8,6])

        # ---------- Theta 1 ----------
        P05 = (T06 @ np.array([0,0,-self.d[5], 1]))[0:3]
        phi1 = np.arctan2(P05[1],P05[0])
        phi2 = np.array([np.arccos(self.d[3]/np.linalg.norm(P05[0:2])), -np.arccos(self.d[3]/np.linalg.norm(P05[0:2]))])

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

        # ---------- Theta 6 ----------
        T60 = np.linalg.inv(T06)
        X60 = T60[0:3,0]
        Y60 = T60[0:3,1]

        for i in range(8):
            theta[i,5] = np.arctan2((-X60[1]*np.sin(theta[i,0])+Y60[1]*np.cos(theta[i,0]))/np.sin(theta[i,4]),
                                    ( X60[0]*np.sin(theta[i,0])-Y60[0]*np.cos(theta[i,0]))/np.sin(theta[i,4]))

        # ------- Theta 3 and 2 -------

        for i in range(8):
            T01 = self.DHLink(self.alpha[0],self.a[0],self.d[0], theta[i,0])
            T45 = self.DHLink(self.alpha[4],self.a[4],self.d[4], theta[i,4])
            T56 = self.DHLink(self.alpha[5],self.a[5],self.d[5], theta[i,5])

            T14 = np.linalg.inv(T01)@T06@np.linalg.inv(T45@T56)
            P14xz = np.array([T14[0,3], T14[2,3]])

            theta[i,2] = np.arccos((np.linalg.norm(P14xz)**2-self.a[1]**2-self.a[2]**2)/(2*self.a[1]*self.a[2]))

            if i % 2 != 0:
                theta[i,2] = -theta[i,2]

            theta[i,1] = np.arctan2(-P14xz[1], -P14xz[0]) - np.arcsin(-self.a[2]*np.sin(theta[i,2])/np.linalg.norm(P14xz))

        # ---------- Theta 4 ----------

        for i in range(8):
            T01 = self.DHLink(self.alpha[0],self.a[0],self.d[0], theta[i,0])
            T12 = self.DHLink(self.alpha[1],self.a[1],self.d[1], theta[i,1])
            T23 = self.DHLink(self.alpha[2],self.a[2],self.d[2], theta[i,2])
            T45 = self.DHLink(self.alpha[4],self.a[4],self.d[4], theta[i,4])
            T56 = self.DHLink(self.alpha[5],self.a[5],self.d[5], theta[i,5])

            T34 = np.linalg.inv(T01@T12@T23)@T06@np.linalg.inv(T45@T56)

            theta[i,3] = np.arctan2(T34[1,0], T34[0,0])

        #print(theta, "\n")
        q = self.nearestQ(theta, last_q)
        return q

'''
np.set_printoptions(suppress=True)
frame = np.array([-0.4385,     -0.1091,     -0.05148,    0.,          0.,          1.57079633])
last_q = np.array([ 0.,         -0.34906585,  1.57079633,  0.34906585, -1.57079633,  0.        ])

T = np.eye(4)
rot = R.from_euler('xyz', frame[3:6])
T[0:3,0:3] = rot.as_matrix()
T[0:3,3] = frame[0:3]

ik = ikSolver()
q = ik.solveIK(T, last_q)
#print(q)

waypoint1 = np.array([-0.4385,     -0.1091,     -0.05148,    0.,          0.,          1.57079633])
waypoint2 = np.array([-0.4385,     0.1091,     -0.05148,    0.,          0.,          1.57079633])

path = np.linspace(waypoint1[0:3],waypoint2[0:3])

UR5 = rtb.models.DH.UR5()

for pose in path:
    T = np.eye(4)
    rot = R.from_euler('xyz', frame[3:6])
    T[0:3,0:3] = rot.as_matrix()
    T[0:3,3] = pose

    #print(T)

    q = ik.solveIK(T, last_q)
    last_q = q
    print(q)

    #print(UR5.fkine(q))

    #print("\n")
'''