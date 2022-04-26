from dmp_cartesian import DMP
import quaternion
import numpy as np

if __name__ == '__main__':
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("demonstration.dat", delimiter=" ", skiprows=1)

    N = 50  # Number of filters
    dmp = DMP(n_bfs=N, alpha_p=48.0, alpha_0=48)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]

    # Fix demonstration, Axis angles switch signs
    demo_axis = demo[:, 0:3]
    for i in range(1, len(demo_axis) - 1):
        if np.dot(demo_axis[i + 1], demo_axis[i]) < 0:
            demo_axis[i + 1] = -1 * demo_axis[i + 1]

    # Convert demonstration orientation to quaternions
    demo_o = quaternion.from_rotation_vector(demo_axis)

    # Ensure that signs don't flip in quaternion space
    for i in range(1, len(demo_o) - 1):
        if np.dot(quaternion.as_float_array(demo_o[i + 1]),quaternion.as_float_array(demo_o[i])) < 0:
            demo_o[i + 1] = -1 * demo_o[i + 1]

    dmp.train(demo_p, demo_o, t, tau)

    tau = 4.648
    # Declare environmental scaling factor for x, y and z
    environment_scaling = np.array([1, 1, 1])
    tNew = np.arange(0, tau * np.max(environment_scaling), 0.002)
    tau *= np.max(environment_scaling)

    # Generate an output trajectory from the trained DMP
    dmp_p, dmp_dp, dmp_ddp, dmp_o, dmp_do, dmp_ddo = dmp.rollout(tau, environment_scaling)

    tp = np.arange(0, len(dmp_p) * 0.002, 0.002)
    to = np.arange(0, len(dmp_o) * 0.002, 0.002)

    #dmp.plot(demo_p, dmp_p, t, tp, y_lable=['X[m]', 'Y[m]', 'Z[m]'], title="Position of TCP")
    dmp.plot(demo_axis, quaternion.as_rotation_vector(dmp_o), t, to, y_lable=['', '', ''], title="Orientation of TCP")





