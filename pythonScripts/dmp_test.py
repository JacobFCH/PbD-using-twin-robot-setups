from __future__ import division, print_function
from dmp_position import PositionDMP
import numpy as np





if __name__ == '__main__':
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("demonstration.dat", delimiter=" ", skiprows=1)

    N = 50 #Number of filters
    dmp = PositionDMP(n_bfs=N, alpha=48.0)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]
    dmp.train(demo_p, t, tau)

    tau /= 2

    dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)  # Generate an output trajectory from the trained DMP

    dmp.plot2DDMP(demo_p, dmp_p, t , t)
    #dmp.plot3DDMP(demo_p,dmp_p)





