from traceback import print_tb
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
    ax[0,0].set_title('Compliant Rotation - X axis')
    ax[0,1].plot(x,position[:,1])
    ax[0,1].set_title('Compliant Rotation - Y axis')
    ax[0,2].plot(x,position[:,2])
    ax[0,2].set_title('Compliant Rotation - Z axis')
    ax[1,0].plot(x,forces[:,0])
    ax[1,0].set_title('External Torque - X axis')
    ax[1,1].plot(x,forces[:,1])
    ax[1,1].set_title('External Torque - Y axis')
    ax[1,2].plot(x,forces[:,2])
    ax[1,2].set_title('External Torque - Z axis')
    plt.show()

# Method to compute the K gain on the quaternion
def comp_kEpsilon(q, K_o):

    s = np.array([[0, -q.z, q.y], [q.z, 0, -q.x],[ -q.y, q.x, 0]])

    e = q.w * np.eye(3) - s

    kPrime = 2 * np.transpose(e)* K_o

    kPrime_oXEpsilon = np.matmul(kPrime, np.array([q.x,q.y,q.z]))

    return kPrime_oXEpsilon

# Method for integarting the quaternions
def q_integration(omega, qt, dt):

    omega = omega * dt/2
    q = quaternion.from_vector_part(omega, vector_axis=-1)
    q_exp = np.exp(q)

    return q_exp * qt

# Method for adding the compliant orientation to the desired orientation and then converting that to axis angles
def quat2axis(o_d, q_epsilon):

    q_d = quaternion.from_rotation_vector(o_d)
    q_c = q_d * q_epsilon
    o_c = quaternion.as_rotation_vector(q_c)
    
    return o_c

# Method for computing the compliant orientation
def compute_oc(o_d, mu, M_o, D_o, K_o, qt, omega, uDo, kEpsilon, dt):
    sum = mu - uDo - kEpsilon
    omega_d = np.matmul(sum,np.linalg.inv(M_o))

    omega = omega + (omega_d * dt)

    q_epsilon = q_integration(omega, qt, dt)

    uDo = np.matmul(omega , D_o)

    kEpsilon = comp_kEpsilon(q_epsilon, K_o)

    o_c = quat2axis(o_d, q_epsilon)

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

    axis_angles = o_d
    qt = quaternion.from_rotation_vector(axis_angles)

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

    print("Plotting Results")
    # Plotting The compliant rotation and the external torques
    o_cs = np.asarray(o_cs)
    torques = np.asarray(torques)

    plot(o_cs,torques)

testController()

