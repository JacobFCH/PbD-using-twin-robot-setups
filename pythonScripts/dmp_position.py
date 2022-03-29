from __future__ import division, print_function

import numpy as np

import matplotlib as mpl
from numpy.ma.core import concatenate
mpl.use('TkAgg') #For interactive plots https://stackoverflow.com/questions/49844189/how-to-get-interactive-plot-of-pyplot-when-using-pycharm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from canonical_system import CanonicalSystem
import scipy


class PositionDMP():
    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None, cs=None):
        self.n_bfs = n_bfs
        self.alpha = alpha
        self.beta = beta if beta is not None else self.alpha / 4
        self.cs = cs if cs is not None else CanonicalSystem(alpha=cs_alpha if cs_alpha is not None else self.alpha/2)

        # Centres of the Gaussian basis functions
        self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c)**2

        # Scaling factor
        self.Dp = np.identity(3)

        # Initially weights are zero (no forcing term)
        self.w = np.zeros((3, self.n_bfs))

        # Initial- and goal positions
        self.p0 = np.zeros(3)
        self.gp = np.zeros(3)

        #Obstacles
        numObstacles = 1
        #pos1 = [-0.322, -0.387, -0.426]
        #pos2 = [-0.686, 0.438, -0.426]
        #xlinspace = np.linspace(pos1[0], pos2[0], 5)
        #ylinspace = np.linspace(pos1[1], pos2[1], 5)
        #zlinspace = np.asarray([pos1[2]]*len(xlinspace))
        #Obstacles first 3 element is the vector and the last is the radius of the sphere.
        self.obstacles = []#[[0.52445584, 0.39893147, 0.41805384, 0.01], [0.661281, 0.156913, 0.382968, 0.02]]#np.random.random((numObstacles, 3)) * 2 - 1
        #for i in range(len(zlinspace)):
        #    for j in range(len(zlinspace)):
        #        self.obstacles.append([xlinspace[i],ylinspace[j],zlinspace[i],0.001])
        self.gamma_o = 2500
        self.gamma_p = 2500
        self.gamma_d = 50
        self.k = 0.01
        self.beta2 = 20/np.pi


        self.reset()

    def step(self, x, dt, tau):
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)





        # Based on https://www.mdpi.com/2076-3417/9/8/1535/html
        def P(y, dy):
            def angleWithDirection(obstacleVector, dy):
                angle = np.arccos(
                    np.dot((obstacleVector).T, dy) / (np.linalg.norm(obstacleVector) * np.linalg.norm(dy)))
                return angle

            def calculatePsi(obstacleVector, dy, angle):
                r = np.cross((obstacleVector), dy)
                r0 = r / np.linalg.norm(r)
                Rv = dy * np.cos(np.pi / 2) + np.cross(r0, dy * np.sin(np.pi / 2)) + np.dot(r0, dy) * r0 * (
                            1 - np.cos(np.pi / 2))

                d = np.linalg.norm(obstacleVector)
                psi = angle * np.exp(-self.beta2 * angle) * np.exp(-self.k * d)
                return psi, Rv

            def nearestPointOnObstacleVector(obstacle, radius, y):
                obstacleVector_p = obstacle - y
                normObstacleVector_p = obstacleVector_p/np.linalg.norm(obstacleVector_p)
                radiusVector = normObstacleVector_p * radius
                return obstacleVector_p - radiusVector

            p_o = np.zeros(3)
            p_p = np.zeros(3)
            p_d = np.zeros(3)
            for obstacleInformation in self.obstacles:
                if np.linalg.norm(dy) > 1e-5:
                    obstacle = obstacleInformation[0:3]
                    radius = obstacleInformation[-1]

                    obstacleVector_o = obstacle - y
                    angle_o = angleWithDirection(obstacleVector_o, dy)
                    psi_o, Rv_o = calculatePsi(obstacleVector_o, dy, angle_o)
                    

                    obstacleVector_p = nearestPointOnObstacleVector(obstacle, radius, y)
                    angle_p = angleWithDirection(obstacleVector_p, dy)
                    psi_p, Rv_p = calculatePsi(obstacleVector_p, dy, angle_p)
                    

                    #if(np.linalg.norm(obstacleVector_p) < 1e-2):
                    #    p_p *= 1000


                    Rv_avg = (Rv_o + Rv_p) * 0.5
                    if not (angle_o > np.pi / 2 or angle_p > np.pi/2):
                        p_o += self.gamma_o * Rv_o * psi_o
                        p_p += self.gamma_p * Rv_p * psi_p
                        p_d += self.gamma_d * Rv_avg * np.exp(-self.k*np.linalg.norm(obstacleVector_p))

            return p_o + p_p + p_d
        # DMP system acceleration
        # TODO: Implement the transformation system differential equation for the acceleration, given that you know the
        # values of the following variables:
        # self.alpha, self.beta, self.gp, self.p, self.dp, tau, x
        self.ddp = self.alpha * (self.beta * (self.gp - self.p) - tau*self.dp) + fp(x) # + P(self.p, self.dp)
        self.ddp /= tau**2


        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        return self.p, self.dp, self.ddp

    def rollout(self, ts, tau):
        self.reset()


        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, 3))
        dp = np.empty((n_steps, 3))
        ddp = np.empty((n_steps, 3))

        for i in range(n_steps):
            p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i])

        return p, dp, ddp

    def reset(self):
        self.p = self.p0.copy()
        self.dp = np.zeros(3)
        self.ddp = np.zeros(3)

    def train(self, positions, ts, tau):
        p = positions

        # Sanity-check input
        if len(p) != len(ts):
            raise RuntimeError("len(p) != len(ts)")

        # Initial- and goal positions
        self.p0 = p[0]
        self.gp = p[-1]

        # Differential time vector
        dt = np.gradient(ts)[:,np.newaxis]

        # Scaling factor
        self.Dp = np.diag(self.gp - self.p0)
        Dp_inv = np.linalg.inv(self.Dp)

        # Desired velocities and accelerations
        d_p = np.gradient(p, axis=0) / dt
        dd_p = np.gradient(d_p, axis=0) / dt

        # Integrate canonical system
        x = self.cs.rollout(ts, tau)

        # Set up system of equations to solve for weights
        def features(xj):
            psi = np.exp(-self.h * (xj - self.c)**2)
            return xj * psi / psi.sum()

        def forcing(j):
            return Dp_inv.dot(tau**2 * dd_p[j]
                - self.alpha * (self.beta * (self.gp - p[j]) - tau * d_p[j]))

        A = np.stack(features(xj) for xj in x)
        f = np.stack(forcing(j) for j in range(len(ts)))

        # Least squares solution for Aw = f (for each column of f)
        self.w = np.linalg.lstsq(A, f, rcond=None)[0].T

        # Cache variables for later inspection
        self.train_p = p
        self.train_d_p = d_p
        self.train_dd_p = dd_p


    def plot2DDMP(self, demo_p,dmp_p, t, tNew):
        # 2D plot the DMP against the original demonstration
        fig1, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(t, demo_p[:, 0], label='Demonstration')
        axs[0].plot(t, dmp_p[:, 0], label='DMP')
        axs[0].set_xlabel('t (s)')
        axs[0].set_ylabel('X (m)')

        axs[1].plot(t, demo_p[:, 1], label='Demonstration')
        axs[1].plot(t, dmp_p[:, 1], label='DMP')
        axs[1].set_xlabel('t (s)')
        axs[1].set_ylabel('Y (m)')

        axs[2].plot(t, demo_p[:, 2], label='Demonstration')
        axs[2].plot(tNew, dmp_p[:, 2], label='DMP')
        axs[2].set_xlabel('t (s)')
        axs[2].set_ylabel('Z (m)')
        axs[2].legend()
        fig1.suptitle("Position of TCP")

        plt.show()

    
    
   
        

    def plot3DDMP(self, demo_p, dmp_p, forceVectors, contactPoints, allPoints, drawObstacles):
        # 3D plot the DMP against the original demonstration
        fig2 = plt.figure(2)
        ax = plt.axes(projection='3d')
        ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
        ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if drawObstacles:
            for obstacle in self.obstacles:
                s = Sphere(ax, x = obstacle[0], y = obstacle[1], z = obstacle[2], radius = obstacle[3])
        if forceVectors.any():
            ax.quiver(forceVectors[0,:],forceVectors[1,:],forceVectors[2,:],forceVectors[6,:],forceVectors[7,:],forceVectors[8,:], length=0.03, color='black', label="Force Vector")
        if contactPoints.any():
            ax.scatter(contactPoints[0,:], contactPoints[1,:], contactPoints[2,:], color='r', label="Contact Points")
        if allPoints.any():
            ax.scatter(allPoints[0,:], allPoints[1,:], allPoints[2,:], color='g',label="No Contact Points")
        ax.legend()
        
        # ax.view_init(30,0)
        plt.show()