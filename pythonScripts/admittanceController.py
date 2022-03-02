from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
import time
import matplotlib.pyplot as plt
import quaternion
import math

class AdmittanceController:
    def __init__(self, dt = 1/500):
        self.dt = dt

        # Positional Parameters
        self.M_p = np.diag([1.0,1.0,1.0])
        self.D_p = np.diag([2.0,2.0,2.0])
        self.K_p = np.diag([1.0,1.0,1.0])
        #self.K_p = np.diag([0.0,0.0,0.0])

        self.pdd_cd = np.array([0.0,0.0,0.0])
        self.pd_cd = np.array([0.0,0.0,0.0])
        self.p_cd = np.array([0.0,0.0,0.0])

        # Rotational Parameters
        self.M_o = np.diag([1.5,1.5,1.5])
        self.D_o = np.diag([6.5,6.5,6.5]) # 6.48074069840786
        self.K_o = np.diag([7.0,7.0,7.0])
        #self.K_o = np.diag([0.0,0.0,0.0])

        self.kEpsilon = np.array([0.0,0.0,0.0])
        self.omega = np.array([0.0,0.0,0.0])
        self.q_epsilon = quaternion.from_rotation_vector([0.0,0.0,0.0])

        # Method to compute the K gain on the quaternion
    def comp_kEpsilon(self, q, K_o):

        s = np.array([[0, -q.z, q.y], [q.z, 0, -q.x],[ -q.y, q.x, 0]])

        e = q.w * np.eye(3) - s

        kPrime = 2 * np.transpose(e)* K_o

        kPrime_oXEpsilon = kPrime @ q.imag

        return kPrime_oXEpsilon

    # Method for computing the compliant orientation
    def computeCompliance(self, d_f, f_t):
        # Compute Positional Compliance
        self.pdd_cd = np.matmul(f_t[0:3] - (self.pd_cd @ self.D_p) - (self.p_cd @ self.K_p) ,np.linalg.pinv(self.M_p))

        self.pd_cd += (self.pdd_cd * self.dt)
        self.p_cd += (self.pd_cd * self.dt)
        p_c = d_f[0:3] + self.p_cd

        # Compute Rotational Complaince
        omega_d = np.matmul(np.linalg.inv(self.M_o), f_t[3:6] - (self.omega @ self.D_o) - self.kEpsilon)

        self.omega += (omega_d * self.dt)
        omega_h = self.omega * self.dt/2

        self.q_epsilon = np.exp(quaternion.quaternion(0,omega_h[0],omega_h[1],omega_h[2])) * self.q_epsilon

        self.kEpsilon = self.comp_kEpsilon(self.q_epsilon, self.K_o)

        eul = R.from_euler('xyz', d_f[3:6])
        d_o = eul.as_rotvec()
        q_c = quaternion.from_rotation_vector(d_o) * self.q_epsilon
        o_c = quaternion.as_rotation_vector(q_c)
        r = R.from_rotvec(o_c)
        o_c = r.as_euler('xyz')

        return np.concatenate((p_c, o_c),axis=0)

    def plot(self, position, forces):
        x = range(len(position))
        fig, ax = plt.subplots(2, 3)
        ax[0,0].plot(x,position[:,0])
        ax[0,0].set_title('Compiant Position - X axis')
        ax[0,1].plot(x,position[:,1])
        ax[0,1].set_title('Compiant Position - Y axis')
        ax[0,2].plot(x,position[:,2])
        ax[0,2].set_title('Compiant Position - Z axis')
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

    def generate_path(self, resolution ,r, x0, y0, z0):
        path = []
        theta = 0
        while theta <= 360:
            x = x0 + r * math.cos(theta * math.pi/180)
            y = y0 + r * math.sin(theta * math.pi/180)
            theta += 360/resolution
            path.append([x,y,z0])
        return path

    def testController(self):
        # Initial Estimates euler zyx
        d_f = np.array([1.0,1.0,1.0,0.0,0.0,np.pi/2])
        f_t = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

        controller = AdmittanceController()

        timestep = 0

        p_cs = []
        o_cs = []
        forces = []
        torques = []
        path = self.generate_path(360, 2, 1 , 1 ,1)
        path_iterator = 0
        # Main loop for testing runs for 5 seconds
        print("Starting Test Loop")
        while timestep < 10:
            #d_f[0:3] = path[path_iterator]
            path_iterator = (path_iterator + 1) % len(path)
            # Computing compliant position using a integrating from 0 to 1, not sure if this is correct
            c_f = controller.computeCompliance(d_f,f_t)
            time.sleep(controller.dt)

            # Adding an external force a 1 second
            if timestep > 2 and timestep < 2 + controller.dt:
                print("adding external force")
                f_t = np.array([1.0,0.0,0.0,0.0,0.0,1.0])

            # Removing the external force at 4 seconds
            if timestep > 5 and timestep < 5 + controller.dt:
                print("no external force")
                f_t = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
            timestep += controller.dt

            p_cs.append(c_f[0:3])
            o_cs.append(c_f[3:6])
            forces.append(f_t[0:3])
            torques.append(f_t[3:6])

        print("Plotting Results")
        # Plotting The compliant postion and the external forces
        p_cs = np.asarray(p_cs)
        o_cs = np.asarray(o_cs)
        forces = np.asarray(forces)
        torques = np.asarray(torques)

        self.plot(p_cs,forces)
        self.plot(o_cs,torques)
        #plot3D(p_cs)