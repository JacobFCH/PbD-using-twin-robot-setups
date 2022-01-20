from traceback import print_tb
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
    ax[0,0].set_title('Compiant Rotation - X axis')
    ax[0,1].plot(x,position[:,1])
    ax[0,1].set_title('Compiant Rotation - Y axis')
    ax[0,2].plot(x,position[:,2])
    ax[0,2].set_title('Compiant Rotation - Z axis')
    ax[1,0].plot(x,forces[:,0])
    ax[1,0].set_title('External Torque - X axis')
    ax[1,1].plot(x,forces[:,1])
    ax[1,1].set_title('External Torque - Y axis')
    ax[1,2].plot(x,forces[:,2])
    ax[1,2].set_title('External Torque - Z axis')
    plt.show()

# Method to compute the K gain on the quaternion
def comp_kEpsilon(q, K_o):

    s = np.array([[0, -q[3], q[2]], [q[3], 0, -q[1]],[ -q[2], q[1], 0]])

    e = q[0] * np.eye(3) - s

    kPrime = 2 * np.transpose(e)* K_o

    kPrime_oXEpsilon = np.matmul(kPrime, np.array([q[1],q[2],q[3]]))

    return kPrime_oXEpsilon

# Method for integarting the quaternions
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

# Method for adding the compliant orientation to the desired orientation and then converting that to a euler angles xyz.
def quat2euler(o_d, q_epsilon):

    rot = R.from_rotvec(o_d)
    q_d = Quaternion(matrix=R.as_matrix(rot))

    q_c = (q_epsilon * q_d.conjugate).conjugate
    o_c = R.from_matrix(q_c.rotation_matrix)
    o_c = o_c.as_euler('xyz')
    
    return o_c

# Method for computing the compliant orientation
def compute_oc(o_d, mu, M_o, D_o, K_o, qt, omega, uDo, kEpsilon, dt):
    sum = mu - uDo - kEpsilon
    omega_d = np.matmul(sum,np.linalg.inv(M_o))

    omega = omega + (omega_d * dt)

    q_epsilon = q_integration(omega, qt, dt)

    uDo = np.matmul(omega , D_o)

    kEpsilon = comp_kEpsilon(q_epsilon, K_o)

    o_c = quat2euler(o_d, q_epsilon)

    return qt, omega, uDo, kEpsilon, o_c

def testController():
    # Initial Estimates
    o_d = np.array([1,1,1])
    mu = np.array([0,0,0])  

    uDo = np.array([0,0,0])
    kEpsilon = np.array([0,0,0])

    omega = np.array([0,0,0])

    M_o = np.array([[1,0,0],[0,1,0],[0,0,1]])
    D_o = np.array([[1,0,0],[0,1,0],[0,0,1]])
    K_o = np.array([[1,0,0],[0,1,0],[0,0,1]])

    euler = o_d
    rot = R.from_euler('xyz',euler)
    qt = Quaternion(matrix=R.as_matrix(rot))

    timestep = 0
    dt = 1/50
    test_duration = 10

    o_cs = []
    torques = []

    # Main loop for testing runs for x seconds
    print("Starting Test")
    while timestep < test_duration:
        qt, omega, uDo, kEpsilon, o_c = compute_oc(o_d, mu, M_o, D_o, K_o, qt, omega, uDo, kEpsilon, dt)
        time.sleep(dt)

        # Adding an external torque a 1 second
        if timestep > 1 and timestep < 1 + dt:
            print("adding external torque")
            mu = np.array([10,-20,30])

        # Removing the external torque at 4 seconds
        if timestep > 4 and timestep < 4 + dt:
            print("no external torque")
            mu = np.array([0,0,0])

        timestep += dt
        o_cs.append(o_c)
        torques.append(mu)

    print("Test Done")
    # Plotting The compliant rotation and the external torques
    o_cs = np.asarray(o_cs)
    torques = np.asarray(torques)

    plot(o_cs,torques)

testController()