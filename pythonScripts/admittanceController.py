from scipy.spatial.transform.rotation import Rotation as R
import matplotlib.pyplot as plt
import numpy as np
import quaternion

class AdmittanceController:
    def __init__(self, dt=1/500, stiffness=False):
        self.dt = dt

        # Positional Parameters
        self.M_p = np.diag([0.7,0.7,0.7])
        self.D_p = np.diag([14,14,14])
        self.K_p = np.diag([5.0,5.0,5.0]) if stiffness else np.diag([0.0,0.0,0.0]) 

        self.pdd_cd = np.array([0.0,0.0,0.0])
        self.pd_cd = np.array([0.0,0.0,0.0])
        self.p_cd = np.array([0.0,0.0,0.0])

        # Rotational Parameters
        self.M_o = np.diag([0.001,0.001,0.001])
        self.D_o = np.diag([0.8, 0.8, 0.2])
        self.K_o = np.diag([7.0,7.0,7.0]) if stiffness else np.diag([0.0,0.0,0.0])

        self.kEpsilon = np.array([0.0,0.0,0.0])
        self.omega = np.array([0.0,0.0,0.0])
        self.q_epsilon = quaternion.from_rotation_vector([0.0,0.0,0.0])

        # Method to compute the K gain on the quaternion
    def comp_kEpsilon(self, q, K_o):

        s = np.array([[0, -q.z, q.y], [q.z, 0, -q.x], [-q.y, q.x, 0]])

        e = q.w * np.eye(3) - s

        kPrime = 2 * np.transpose(e) * K_o

        kPrime_oXEpsilon = kPrime @ q.imag

        return kPrime_oXEpsilon

    # Method for computing the compliant orientation
    def computeCompliance(self, d_f, f_t, rotation):
        # Compute Positional Compliance
        self.pdd_cd = np.matmul(f_t[0:3] - (self.pd_cd @ self.D_p) - (self.p_cd @ self.K_p), np.linalg.pinv(self.M_p))

        self.pd_cd += (self.pdd_cd * self.dt)
        self.p_cd += (self.pd_cd * self.dt)
        p_c = d_f[0:3] + (rotation @ self.p_cd)

        # Compute Rotational Complaince
        omega_d = np.matmul(np.linalg.inv(self.M_o), f_t[3:6] - (self.omega @ self.D_o) - self.kEpsilon)

        self.omega += (omega_d * self.dt)
        omega_h = self.omega * self.dt/2
        self.q_epsilon = np.exp(quaternion.quaternion(0,omega_h[0],omega_h[1],omega_h[2])) * self.q_epsilon

        self.kEpsilon = self.comp_kEpsilon(self.q_epsilon, self.K_o)

        d_o = d_f[3:6]
        q_c = quaternion.from_rotation_vector(d_o) * self.q_epsilon
        o_c = quaternion.as_rotation_vector(q_c)

        return np.concatenate((p_c, o_c),axis=0)

    def plot(self, position, forces):
        x = range(len(position))
        fig, ax = plt.subplots(2, 3)
        ax[0,0].plot(x,position[:,0])
        ax[0,0].set_title('Compliant Position - X axis')
        ax[0,1].plot(x,position[:,1])
        ax[0,1].set_title('Compliant Position - Y axis')
        ax[0,2].plot(x,position[:,2])
        ax[0,2].set_title('Compliant Position - Z axis')
        ax[1,0].plot(x,forces[:,0])
        ax[1,0].set_title('External Force - X axis')
        ax[1,1].plot(x,forces[:,1])
        ax[1,1].set_title('External Force - Y axis')
        ax[1,2].plot(x,forces[:,2])
        ax[1,2].set_title('External Force - Z axis')
        plt.show()

    def plot3D(self, position):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(position[:,0],position[:,1],position[:,2])
        plt.show()