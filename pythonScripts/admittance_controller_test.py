import re
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
import time
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
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

def comp_kEpsilon(q, K_o):

    s = np.array([[0, -q[3], q[2]], [q[3], 0, -q[1]],[ -q[2], q[1], 0]])

    e = q[0] * np.eye(3) - s

    kPrime = 2 * np.transpose(e)* K_o

    kPrime_oXEpsilon = np.matmul(kPrime, np.array([q[1],q[2],q[3]]))

    return kPrime_oXEpsilon

def q_integration(omega, qt, dt):

    omega = omega * dt/2

    if qt.norm > 0:
        norm_omega = np.linalg.norm(omega)
        if norm_omega > 0:
            epsilon = (omega/norm_omega)*math.sin(norm_omega)
            eta = math.cos(norm_omega)
            exp = Quaternion(np.array([eta, epsilon[0], epsilon[1], epsilon[2]]))
            q = exp*qt
        else:
            q = qt
    else:
        q = qt   

    q = q.normalised
    return q

def quat2rotvec(o_d, q_epsilon):

    rot = R.from_rotvec(o_d)
    q_d = Quaternion(matrix=R.as_matrix(rot))

    q_c = (q_epsilon * q_d.conjugate).conjugate
    o_c = R.from_matrix(q_c.rotation_matrix)
    o_c = o_c.as_rotvec()

    return o_c

# Method for computing the compliant positon at each timestep
def compute_pc(d_p, h_e, uD_p, uK_p, pdd_cd, pd_cd, p_cd, M_p, D_p, K_p, dt):
    sum = np.asarray(h_e - uD_p - uK_p)
    pdd_cd = np.matmul(sum,np.linalg.pinv(M_p))

    pd_cd = pd_cd + (pdd_cd * dt)

    p_cd = p_cd + (pd_cd * dt)

    p_c = d_p + p_cd

    uD_p = np.matmul(pd_cd , D_p)
    uK_p = np.matmul(p_cd, K_p)

    return pdd_cd, pd_cd, p_cd, uD_p, uK_p, p_c

def generate_path(resolution ,r, x0, y0, z0):
    path = []
    theta = 0
    while theta <= 360:
        x = x0 + r * math.cos(theta * math.pi/180)
        y = y0 + r * math.sin(theta * math.pi/180)
        theta += 360/resolution
        path.append([x,y,z0])
    return path

def compute_oc(o_d, mu, M_o, D_o, K_o, qt, omega, uDo, kEpsilon, dt):
    sum = mu - uDo - kEpsilon
    omega_d = np.matmul(sum,np.linalg.inv(M_o))

    omega = omega + (omega_d * dt)

    q_epsilon = q_integration(omega, qt, dt)

    uDo = np.matmul(omega , D_o)

    kEpsilon = comp_kEpsilon(q_epsilon, K_o)

    o_c = quat2rotvec(o_d, q_epsilon)

    return qt, omega, uDo, kEpsilon, o_c

def testController():
    # Initial Estimates
    d_p = np.array([1,1,1])
    o_d = np.array([0,0,0])
    h_e = np.array([0,0,0]) 
    mu = np.array([0,0,0])  

    uD_p = np.array([0,0,0])
    uK_p = np.array([0,0,0])

    uDo = np.array([0,0,0])
    kEpsilon = np.array([0,0,0])

    pdd_cd = np.array([0,0,0])
    pd_cd = np.array([0,0,0])
    p_cd = np.array([0,0,0])

    omega = np.array([0,0,0])

    M_p = np.array([[1,0,0],[0,1,0],[0,0,1]])
    D_p = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K_p = np.array([[1,0,0],[0,1,0],[0,0,1]])

    M_o = np.array([[1,0,0],[0,1,0],[0,0,1]])
    D_o = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K_o = np.array([[1,0,0],[0,1,0],[0,0,1]])

    rotvec = o_d
    rot = R.from_rotvec(rotvec)
    qt = Quaternion(matrix=R.as_matrix(rot))

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
        pdd_cd, pd_cd, p_cd, uD_p, uK_p, p_c = compute_pc(d_p, h_e, uD_p, uK_p, pdd_cd, pd_cd, p_cd, M_p, D_p, K_p, dt)
        qt, omega, uDo, kEpsilon, o_c = compute_oc(o_d, mu, M_o, D_o, K_o, qt, omega, uDo, kEpsilon, dt)
        time.sleep(dt)

        # Adding an external force a 1 second
        if timestep > 5 and timestep < 5 + dt:
            print("adding external force")
            h_e = np.array([-1,0,0])
            mu = np.array([0,0,0])

        # Removing the external force at 4 seconds
        if timestep > 8 and timestep < 8 + dt:
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

    #plot(p_cs,forces)
    #plot(o_cs,torques)
    plot3D(p_cs)

testController()