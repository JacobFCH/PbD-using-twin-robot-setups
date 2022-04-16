from dmp_cartesian import DMP
import quaternion
import numpy as np

if __name__ == '__main__':
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("demonstration.dat", delimiter=" ", skiprows=1)

    N = 50 #Number of filters
    dmp = DMP(n_bfs=N, alpha=48.0)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]

    # Convert demonstration orientation to quaternions
    demo_o = quaternion.from_euler_angles(demo[:, 3:6])

    dmp.train(demo_p, demo_o, t, tau)

    tau /= 1

    dmp_p, dmp_dp, dmp_ddp, dmp_o, dmp_do, dmp_ddo = dmp.rollout(t, tau)  # Generate an output trajectory from the trained DMP

    #dmp.plot2DDMP_Position(demo_p, dmp_p, t , t)
    dmp.plot2DDMP_Orientation(demo[:, 3:6], quaternion.as_euler_angles(dmp_o), t , t)





