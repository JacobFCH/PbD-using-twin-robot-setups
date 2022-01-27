from traceback import print_tb
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
import time
import matplotlib.pyplot as plt
import quaternion
import math

# TO DO

# Change the plots to be on the quaternions and the forces
# Check all the individual steps to ensure that i get what i expect
# Find out why some of the quaternions are doing stupid stuff
# The core problem might be the quaternion that is created from omega_half, or the quaternion resulting from the rotation around the desired frame
# Check dimensions on everything and ensure that they are correct, and make sure numpy doesnt do stupid broadcasting


def plotQuat(quat_list, forces):
    x = range(len(quat_list))
    fig, ax = plt.subplots(2, 4)
    ax[0,0].plot(x,quat_list[:,0])
    ax[0,0].set_title('Quat Omega')
    ax[0,1].plot(x,quat_list[:,1])
    ax[0,1].set_title('Quat i')
    ax[0,2].plot(x,quat_list[:,2])
    ax[0,2].set_title('Quat j')
    ax[0,3].plot(x,quat_list[:,3])
    ax[0,3].set_title('Quat k')
    ax[1,0].plot(x,forces[:,0])
    ax[1,0].set_title('External Torque - X axis')
    ax[1,1].plot(x,forces[:,1])
    ax[1,1].set_title('External Torque - Y axis')
    ax[1,2].plot(x,forces[:,2])
    ax[1,2].set_title('External Torque - Z axis')
    plt.show()

def plot(position, forces):
    x = range(len(position))
    fig, ax = plt.subplots(2, 3)
    ax[0,0].plot(x,position[:,0])
    ax[0,0].set_title('Compiant Orientation - X axis')
    ax[0,1].plot(x,position[:,1])
    ax[0,1].set_title('Compiant Orientation - Y axis')
    ax[0,2].plot(x,position[:,2])
    ax[0,2].set_title('Compiant Orientation - Z axis')
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

def testController():
    # Initial Estimates
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
    test_duration = 10

    o_cs = []
    torques = []

    # Main loop for testing runs for x seconds
    print("Starting Test")
    while timestep < test_duration:
        q_epsilon, omega, kEpsilon, o_c = compute_oc(o_d, mu, M_o, D_o, K_o, q_epsilon, omega, kEpsilon, dt)
        time.sleep(dt)

        # Adding an external torque a 1 second
        if timestep > 1 and timestep < 1 + dt:
            print("adding external torque")
            mu = np.array([10,0,0])

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

