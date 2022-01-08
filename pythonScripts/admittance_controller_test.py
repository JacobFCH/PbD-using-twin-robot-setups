from scipy.integrate import quad
import numpy as np
import time
import matplotlib.pyplot as plt

# Method for computing the compliant positon at each timestep
def TestImport():
    print("Yes")

def compute_Comliance(d_p, h_e, uD_p, uK_p, pdd_cd, pd_cd, p_cd, M_p, D_p, K_p, timestep):
    sum = np.asarray(h_e - uD_p - uK_p)
    pdd_cd = np.matmul(sum,np.linalg.pinv(M_p))

    pd_cd = pd_cd + (pdd_cd * timestep)

    p_cd = p_cd + (pd_cd * timestep)

    p_c = d_p + p_cd

    uD_p = np.matmul(pd_cd , D_p)
    uK_p = np.matmul(p_cd, K_p)

    return pdd_cd, pd_cd, p_cd, uD_p, uK_p, p_c

def testController():
    # Initial Estimates
    d_p = np.array([1,1,1])
    h_e = np.array([0,0,0])  

    uD_p = np.array([0,0,0])
    uK_p = np.array([0,0,0])

    pdd_cd = np.array([0,0,0])
    pd_cd = np.array([0,0,0])
    p_cd = np.array([0,0,0])

    M_p = np.array([[1,0,0],[0,1,0],[0,0,1]])
    D_p = np.array([[10,0,0],[0,10,0],[0,0,10]])
    K_p = np.array([[0,0,0],[0,0,0],[0,0,0]])

    timestep = 0
    dt = 1/50

    p_cs = []
    forces = []

    # Main loop for testing runs for 5 seconds
    while timestep < 10:
        # Computing compliant position using a integrating from 0 to 1, not sure if this is correct
        pdd_cd, pd_cd, p_cd, uD_p, uK_p, p_c = compute_Comliance(d_p, h_e, uD_p, uK_p, pdd_cd, pd_cd, p_cd, M_p, D_p, K_p, dt)
        time.sleep(dt)

        # Adding an external force a 1 second
        if timestep > 1 and timestep < 1 + dt:
            print("adding external force")
            h_e = np.array([1,0,0])

        # Removing the external force at 4 seconds
        if timestep > 4 and timestep < 4 + dt:
            print("no external force")
            h_e = np.array([0,0,0]) 

        timestep += dt
        p_cs.append(p_c)
        forces.append(h_e)

    # Plotting The compliant postion and the external forces
    p_cs = np.asarray(p_cs)
    forces = np.asarray(forces)

    x = range(len(p_cs))
    fig, ax = plt.subplots(2, 3)
    ax[0,0].plot(x,p_cs[:,0])
    ax[0,0].set_title('Compiant Position - X axis')
    ax[0,1].plot(x,p_cs[:,1])
    ax[0,1].set_title('Compiant Position - Y axis')
    ax[0,2].plot(x,p_cs[:,2])
    ax[0,2].set_title('Compiant Position - Z axis')
    ax[1,0].plot(x,forces[:,0])
    ax[1,0].set_title('External Force - X axis')
    ax[1,1].plot(x,forces[:,1])
    ax[1,1].set_title('External Force - Y axis')
    ax[1,2].plot(x,forces[:,2])
    ax[1,2].set_title('External Force - Z axis')
    plt.show()