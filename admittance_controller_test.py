from scipy.integrate import quad
import numpy as np
import time
import matplotlib.pyplot as plt

# Integrator and integrad function, not sure if this is correct
def integrand(x, a):
    return x*a

def integrator(mat, timestep):
    ret = np.zeros((3))
    for i in range(mat.shape[0]):
        I = quad(integrand,0,timestep, args=(mat[i]))
        ret[i] = I[0]

    return ret 

# Method for computing the compliant positon at each timestep
def compute_Comliance(desired_position, external_force, uD_p, uK_p, M_p, D_p, K_p, timestep):
    sum = np.asarray(external_force - uD_p - uK_p)
    pdd_cd = np.matmul(sum,np.linalg.pinv(M_p))

    pd_cd = integrator(pdd_cd,timestep)

    uD_p = pd_cd * D_p

    p_cd = integrator(pd_cd,timestep)

    uK_p = p_cd * K_p

    p_c = desired_position + p_cd

    uD_p = np.matmul(pd_cd , D_p)
    uK_p = np.matmul(p_cd, K_p)

    return uD_p, uK_p, p_c

# Initial Estimates
desired_position = np.array([1,1,1])
external_force = np.array([0,0,0])  

uD_p = np.array([0,0,0])
uK_p = np.array([0,0,0])

M_p = np.array([[1,0,0],[0,1,0],[0,0,1]])
D_p = np.array([[1,0,0],[0,1,0],[0,0,1]])
K_p = np.array([[1,0,0],[0,1,0],[0,0,1]])

timestep = 0
hz = 0.02

p_cs = []
forces = []

# Main loop for testing runs for 5 seconds
while timestep < 5:
    # Computing compliant position using a integrating from 0 to 1, not sure if this is correct
    uD_p, uK_p, p_c = compute_Comliance(desired_position, external_force, uD_p, uK_p, M_p, D_p, K_p, 1)
    time.sleep(hz)

    # Adding an external force a 1 second
    if timestep > 1 and timestep < 1 + hz:
        print("adding external force")
        external_force = np.array([1,0,0])

    # Removing the external force at 4 seconds
    if timestep > 4 and timestep < 4 + hz:
        print("no external force")
        external_force = np.array([0,0,0]) 

    timestep += hz
    p_cs.append(p_c)
    forces.append(external_force)

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