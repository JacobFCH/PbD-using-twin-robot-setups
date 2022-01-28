import re
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
import time
import matplotlib.pyplot as plt
import quaternion
import math

def plot(position, forces):
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

def plot3D(position):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(position[:,0],position[:,1],position[:,2])
    plt.show()


# Method to compute the K gain on the quaternion
def comp_kEpsilon(q, K_o):

    s = np.array([[0, -q.z, q.y], [q.z, 0, -q.x],[ -q.y, q.x, 0]])

    e = q.w * np.eye(3) - s

    kPrime = 2 * np.transpose(e)* K_o

    kPrime_oXEpsilon = kPrime @ q.imag

    return kPrime_oXEpsilon

# Method for computing the compliant orientation
def compute_oc(o_d, mu, M_o, D_o, K_o, qt, omega, kEpsilon, dt):
    omega_d = np.matmul(np.linalg.inv(M_o), mu - (omega @ D_o) - kEpsilon)

    omega += (omega_d * dt)
    omega_h = omega * dt/2

    q_epsilon = np.exp(quaternion.quaternion(0,omega_h[0],omega_h[1],omega_h[2])) * qt

    kEpsilon = comp_kEpsilon(q_epsilon, K_o)

    q_c = quaternion.from_rotation_vector(o_d) * q_epsilon
    o_c = quaternion.as_rotation_vector(q_c)

    return q_epsilon, omega, kEpsilon, o_c


# Method for computing the compliant positon at each timestep
def compute_pc(d_p, h_e, pdd_cd, pd_cd, p_cd, M_p, D_p, K_p, dt):
    pdd_cd = np.matmul(h_e - (pd_cd @ D_p) - (p_cd @ K_p) ,np.linalg.pinv(M_p))

    pd_cd = pd_cd + (pdd_cd * dt)
    p_cd = p_cd + (pd_cd * dt)
    p_c = d_p + p_cd

    return pdd_cd, pd_cd, p_cd, p_c

def generate_path(resolution ,r, x0, y0, z0):
    path = []
    theta = 0
    while theta <= 360:
        x = x0 + r * math.cos(theta * math.pi/180)
        y = y0 + r * math.sin(theta * math.pi/180)
        theta += 360/resolution
        path.append([x,y,z0])
    return path

def testController():
    # Initial Estimates
    d_p = np.array([1.0,1.0,1.0])
    h_e = np.array([0.0,0.0,0.0])

    pdd_cd = np.array([0.0,0.0,0.0])
    pd_cd = np.array([0.0,0.0,0.0])
    p_cd = np.array([0.0,0.0,0.0])

    M_p = np.diag([1.0,1.0,1.0])
    D_p = np.diag([2.0,2.0,2.0])
    K_p = np.diag([1.0,1.0,1.0])
    #K_p = np.diag([0.0,0.0,0.0])

    o_d = np.array([np.pi/2,0.0,0.0])
    mu = np.array([0.0,0.0,0.0])  

    kEpsilon = np.array([0.0,0.0,0.0])

    omega = np.array([0.0,0.0,0.0])

    M_o = np.diag([1.5,1.5,1.5])
    D_o = np.diag([6.48074069840786,6.48074069840786,6.48074069840786])
    K_o = np.diag([7.0,7.0,7.0])
    #K_o = np.diag([0.0,0.0,0.0])

    axis_angles = [0.0,0.0,0.0]
    q_epsilon = quaternion.from_rotation_vector(axis_angles)

    timestep = 0
    dt = 1/50

    p_cs = []
    o_cs = []
    forces = []
    torques = []
    path = generate_path(360, 2, 1 , 1 ,1)
    path_iterator = 0
    # Main loop for testing runs for 5 seconds
    print("Starting Test Loop")
    while timestep < 20:
        d_p = path[path_iterator]
        path_iterator = (path_iterator + 1) % len(path)
        # Computing compliant position using a integrating from 0 to 1, not sure if this is correct
        pdd_cd, pd_cd, p_cd, p_c = compute_pc(d_p, h_e, pdd_cd, pd_cd, p_cd, M_p, D_p, K_p, dt)
        q_epsilon, omega, kEpsilon, o_c = compute_oc(o_d, mu, M_o, D_o, K_o, q_epsilon, omega, kEpsilon, dt)
        time.sleep(dt)

        # Adding an external force a 1 second
        if timestep > 2 and timestep < 2 + dt:
            print("adding external force")
            h_e = np.array([1,0,0])
            mu = np.array([0,0,0])

        # Removing the external force at 4 seconds
        if timestep > 5 and timestep < 5 + dt:
            print("no external force")
            h_e = np.array([0,0,0])
            mu = np.array([0,0,0])

        timestep += dt
        p_cs.append(p_c)
        o_cs.append(o_c)
        forces.append(h_e)
        torques.append(mu)

    print("Plotting Results")
    # Plotting The compliant postion and the external forces
    p_cs = np.asarray(p_cs)
    o_cs = np.asarray(o_cs)
    forces = np.asarray(forces)
    torques = np.asarray(torques)

    plot(p_cs,forces)
    #plot(o_cs,torques)
    plot3D(p_cs)

testController()