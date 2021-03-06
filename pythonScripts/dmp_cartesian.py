import numpy as np
import quaternion
import matplotlib as mpl
from numpy.ma.core import concatenate
mpl.use('TkAgg') #For interactive plots https://stackoverflow.com/questions/49844189/how-to-get-interactive-plot-of-pyplot-when-using-pycharm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from canonical_system import CanonicalSystem
from scipy.linalg import logm
from rotodilatation import compute_rotodilation

import time


class DMP():
    def __init__(self, n_bfs=10, alpha_p=48, alpha_0=48, beta=None, cs_alpha=None, cs=None):
        self.n_bfs = n_bfs
        self.alpha_p = alpha_p
        self.beta_p = beta if beta is not None else self.alpha_p / 4

        self.alpha_o = alpha_0
        self.beta_o = beta if beta is not None else self.alpha_o / 4

        self.cs = cs if cs is not None else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else self.alpha_p/2)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c)**2

        # Scaling factor
        self.Dp = np.identity(3)
        self.Do = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w_p = np.zeros((3, self.n_bfs))
        self.w_o = np.zeros((3, self.n_bfs))

        # Initial- and goal positions
        self.p0 = np.zeros(3)
        self.gp = np.zeros(3)

        self.omega_do = np.zeros(3)
        self.omega_o = np.zeros(3)

        self.o0 = quaternion.from_float_array([0,0,0,0])
        self.go = quaternion.from_float_array([0,0,0,0])

        self.do = np.zeros(3)
        self.o = quaternion.from_float_array([0,0,0,0])

        self.K = self.alpha_p * self.beta_p
        self.D = self.alpha_p

        self.dt = 0.002

        self.S = np.eye(3)

        self.max_acc = np.array([np.inf, np.inf, np.inf])
        self.max_vel = np.array([np.inf, np.inf, np.inf])
        self.gamma_a = 1
        self.gamma_nom = 1
        self.epsilon = 0.001

        self.tau_nom = 0

        self.reset()

    def setConstraints(self, vel, acc):
        self.max_vel = vel
        self.max_acc = acc

    def compute_scaling(self, p0, gp, environment_scaling):
        # Compute scaling term S
        # Compute the new start and goal position
        p0_prime = p0 * environment_scaling
        gp_prime = gp * environment_scaling

        # Compute normalized vectors based on the new goal and starting position
        gp_x0 = gp - p0
        gp_prime_x0_prime = gp_prime - p0_prime

        # Compute the rotodialtion mapping the vector gp - p0 to the vector gp_prime - p0_prime
        return compute_rotodilation(gp_x0, gp_prime_x0_prime), p0_prime, gp_prime

    def setup_matrices(self, dp, ddp, tau, S):

        x = self.cs.predict_step(self.dt, tau)
        _, dp_next, ddp_next = self.predict_step(x, self.dt, tau, S)

        # Matrices representing the velocity and acceleration
        Ak = np.array([-dp, dp]) * tau
        Ck = np.array([ddp, -ddp]) * tau**2

        # Matrices containing the max acceleration and max velocity
        Ba = np.array([-self.max_acc, -self.max_acc])
        Dv = np.array([-self.max_vel, -self.max_vel])

        # Matrices representing the velocity and acceleration of the next timestep
        Ak_next = np.array([-dp_next, dp_next]) * tau
        Ck_next = np.array([-ddp_next, ddp_next]) * tau**2

        return Ak, Ak_next, Ba, Ck, Ck_next, Dv

    def time_coupling(self, tau, dp, ddp):
        # Based on https://github.com/albindgit/TC_DMP_constrainedVelAcc

        Ak, Ak_next, Ba, Ck, Ck_next, Dv = self.setup_matrices(dp, ddp, tau, self.S)

        tau_min_a = np.max(-(Ba[Ak < 0] * (tau ** 2) + Ck[Ak < 0]) / Ak[Ak < 0])
        tau_max_a = np.min(-(Ba[Ak > 0] * (tau ** 2) + Ck[Ak > 0]) / Ak[Ak > 0])

        tau_min_v = (np.max(- Ak_next / Dv) - tau) / self.dt

        Ais = Ak[Ak < 0]
        Ajs = Ak[Ak > 0]
        Bis = Ba[Ak < 0]
        Bjs = Ba[Ak > 0]
        Cis = Ck[Ak < 0]
        Cjs = Ck[Ak > 0]
        tauNext = -np.inf
        for i in range(len(Ais)):
            for j in range(len(Ajs)):
                num = Cis[i] * np.abs(Ajs[j]) + Cjs[j] * np.abs(Ais[i])
                den = np.abs(Bis[i] * Ajs[j]) + np.abs(Bjs[j] * Ais[i])
                if num > 0:
                        if np.sqrt(num/den) > tauNext:
                            tauNext = np.sqrt(num/den)
        tau_min_f = (tauNext - tau) /self.dt

        tau_min_nom = (self.tau_nom - tau) / self.dt

        tau_min = max(tau_min_a, tau_min_v, tau_min_f, tau_min_nom)

        y_dotdot = (ddp * tau**2) / ((tau ** 2) * self.max_acc)

        sigma_y = 0
        for i in range(len(y_dotdot)):
            sigma_y += y_dotdot[i] ** 2 / max(1 - y_dotdot[i] ** 2, self.gamma_a * self.epsilon)
        sigma_y *= self.gamma_a

        tau_hat = self.gamma_nom * (self.tau_nom - tau) + tau * sigma_y

        tau_dot = max(min(tau_hat, tau_max_a), tau_min)

        return tau_dot

    def predict_step(self, x, dt, tau, S):
        # -------------------- Positional DMP step --------------------

        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return self.Dp.dot(self.w_p.dot(psi) / psi.sum() * xj)

        ddp = self.K * (self.gp - self.p) - self.D * (tau * self.dp) + np.dot(S, fp(x))
        ddp /= tau ** 2

        # Integrate acceleration to obtain velocity
        dp = self.dp + self.ddp * dt

        # Integrate velocity to obtain position
        p = self.p + self.dp * dt
        return p, dp, ddp

    def step(self, x, dt, tau, S):
        # -------------------- Positional DMP step --------------------

        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Dp.dot(self.w_p.dot(psi) / psi.sum() * xj)

        self.ddp = self.K * (self.gp - self.p) - self.D * (tau * self.dp) + np.dot(S, fp(x))
        self.ddp /= tau**2

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        # -------------------- Rotational DMP Step --------------------

        def fo(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Do.dot(self.w_o.dot(psi) / psi.sum() * xj)

        self.ddo = self.alpha_o * (self.beta_o * 2 * np.log(self.go * self.o.conjugate()).vec - tau*self.do) + fo(x)
        self.ddo /= tau**2

        # Integrate acceleration to obtain velocity
        self.do += self.ddo * dt
        self.o = np.exp(dt / 2 * quaternion.quaternion(0, *self.do)) * self.o

        return self.p, self.dp, self.ddp, self.o, self.do, self.ddo

    def rollout(self, tau, environment_scaling):
        self.reset()

        p = np.array([[0., 0., 0.]])
        dp = np.array([[0., 0., 0.]])
        ddp = np.array([[0., 0., 0.]])

        o = np.array([])
        do = np.array([])
        ddo = np.array([])

        phase = np.array([])

        err = np.nan_to_num(np.inf)
        tol = 0.001
        i = 0

        self.S, self.p, self.gp = self.compute_scaling(self.p0, self.gp, environment_scaling)

        self.tau_nom = tau
        tau_k = tau

        self.cs.reset()
        while err > tol:
            x = self.cs.step(self.dt, tau_k)
            phase = np.append(phase, x)
            p_element, dp_element, ddp_element, o_element, do_element, ddo_element = self.step(x, self.dt, tau_k, self.S)
            p, dp, ddp = np.append(p, [p_element], axis=0), np.append(dp, [dp_element], axis=0), np.append(ddp, [ddp_element], axis=0)
            o, do, ddo = np.append(o, o_element), np.append(do, do_element) , np.append(ddo, ddo_element)

            err = np.linalg.norm(np.abs(p_element) - np.abs(self.gp))
            i += 1

            tau_dot = self.time_coupling(tau_k, dp_element, ddp_element)
            tau_k = tau_k + tau_dot * self.dt

        return p[1:-1], dp[1:-1], ddp[1:-1], o, do, ddo, phase[0:-1]

    def reset(self):
        self.p = self.p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)

        self.o = self.o0
        self.do = np.zeros(3)
        self.ddo = quaternion.from_float_array([0,0,0,0])

    def train(self, positions, orientations, ts, tau):
        p = positions
        o = orientations

        # Sanity-check position and orientation
        if len(p) != len(ts):
            raise RuntimeError("len(p) != len(ts)")
        if len(o) != len(ts):
            raise RuntimeError("len(o) != len(ts)")

        # Initial- and goal positions and orientations
        self.p0 = p[0]
        self.gp = p[-1]

        self.o0 = o[0]
        self.go = o[-1]

        # Differential time vector
        dt = np.gradient(ts)[:,np.newaxis]

        # Scaling factor
        self.Dp = np.diag(self.gp - self.p0)
        Dp_inv = np.linalg.inv(self.Dp)

        self.Do = np.diag(np.diag(logm(quaternion.as_rotation_matrix(self.go) @ np.transpose(quaternion.as_rotation_matrix(self.o0)))))
        Do_inv = np.linalg.inv(self.Do)

        # Desired velocities and accelerations
        d_p = np.gradient(p, axis=0) / dt
        dd_p = np.gradient(d_p, axis=0) / dt

        d_o = np.empty([len(o),3])
        dd_o = np.empty([len(o),3])

        for i in range(1, len(o) - 1):
            d_o[i] = quaternion.as_vector_part(2 * np.log(o[i + 1] * o[i].conjugate()))
        dd_o = np.gradient(d_o, axis=0) / dt

        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return xj * psi / psi.sum()

        def forcing_p(j):
            return Dp_inv.dot(tau**2 * dd_p[j] - self.alpha_p * (self.beta_p * (self.gp - p[j]) - tau * d_p[j]))

        # np.log(self.go - o[j].conjugate())
        def forcing_o(j):
            return Do_inv.dot((tau**2 * dd_o[j]) - (self.alpha_o * (self.beta_o * 2 * quaternion.as_vector_part(np.log(self.go * o[j].conjugate()))) - (tau * d_o[j])))

        A = np.stack(features(xj) for xj in x)
        f_p = np.stack(forcing_p(j) for j in range(len(ts)))
        f_o = np.stack(forcing_o(j) for j in range(len(ts)))

        # Least squares solution for Aw = f (for each column of f)
        self.w_p = np.linalg.lstsq(A, f_p, rcond=None)[0].T
        self.w_o = np.linalg.lstsq(A, f_o, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_p = p
        self.train_d_p = d_p
        self.train_dd_p = dd_p

    def plot(self, demo, dmp, t, t_dmp, y_label=['', '', ''], title="DMP", plot_demo=True):

        max_acc = np.repeat(0.2, len(t_dmp))
        min_acc = np.repeat(-0.2, len(t_dmp))

        # 2D plot the DMP against the original demonstration
        fig1, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(t_dmp, dmp[:, 0], label='Constrained DMP')
        if plot_demo: axs[0].plot(t, demo[:, 0], '--', label='Unconstrained DMP')
        axs[0].plot(t_dmp, min_acc, '--', label='Min velocity')
        axs[0].plot(t_dmp, max_acc, '--', label='Max velocity')
        axs[0].set_xlabel('t (s)')
        axs[0].set_ylabel(y_label[0])

        axs[1].plot(t_dmp, dmp[:, 1], label='Constrained DMP')
        if plot_demo: axs[1].plot(t, demo[:, 1], '--', label='Unconstrained DMP')
        axs[1].plot(t_dmp, min_acc, '--', label='Min velocity')
        axs[1].plot(t_dmp, max_acc, '--', label='Max velocity')
        axs[1].set_xlabel('t (s)')
        axs[1].set_ylabel(y_label[1])

        axs[2].plot(t_dmp, dmp[:, 2], label='Constrained DMP')
        if plot_demo: axs[2].plot(t, demo[:, 2], '--', label='Unconstrained DMP')
        axs[2].plot(t_dmp, min_acc, '--', label='Min velocity')
        axs[2].plot(t_dmp, max_acc, '--', label='Max velocity')
        axs[2].set_xlabel('t (s)')
        axs[2].set_ylabel(y_label[2])
        axs[2].legend()
        fig1.suptitle(title)

        plt.show()

    def plot3DDMP(self, demo_p, dmp_p, plot_demo=True):
        # 3D plot the DMP against the original demonstration
        fig2 = plt.figure(2)
        ax = plt.axes(projection='3d')
        ax.scatter(dmp_p[0, 0], dmp_p[0, 1], dmp_p[0, 2], color="green")
        ax.scatter(dmp_p[-1, 0], dmp_p[-1, 1], dmp_p[-1, 2], color="red")
        ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='Unconstrained DMP')
        if plot_demo: ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Constrained DMP')
        ax.legend()
        ax.set_xlabel('X[m]')
        ax.set_ylabel('Y[m]')
        ax.set_zlabel('Z[m]')

        plt.show()