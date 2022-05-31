from dmp_cartesian import DMP
import quaternion
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation as R

if __name__ == '__main__':
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("demonstration.dat", delimiter=" ", skiprows=1)
    demo_2 = np.loadtxt("virtual_demo.dat", delimiter=" ", skiprows=1)

    N = 50  # Number of filters
    dmp = DMP(n_bfs=N, alpha_p=48.0, alpha_0=8*48)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]
    demo_v = demo_2[:, 0:3]

    # Fix demonstration, Axis angles switch signs, Remember to remove this is the new demo looks good

    demo_axis = demo[:, 0:3]

    test_pls = np.linspace(np.array([0.1,0.2,0.1]),np.array([1.2,0.6,0.1]), len(demo_axis))
    for i in range(len(test_pls)):
        r = R.from_euler('xyz',test_pls[i])
        test_pls[i] = r.as_rotvec()

    demo_axis = test_pls

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

    # Declare environmental scaling factor for x, y and z
    environment_scaling = 1
    tNew = np.arange(0, tau * environment_scaling, 0.002)
    #tau *= environment_scaling

    # Pre constraint DMP
    dmp_p_pre, dmp_dp_pre, dmp_ddp_pre, dmp_o_pre, dmp_do_pre, dmp_ddo_pre, phase_pre = dmp.rollout(tau, environment_scaling)

    max_acc = np.array([np.inf, np.inf, np.inf])
    #max_acc = np.array([1, 1, 1])
    max_vel = np.array([0.2, 0.2, 0.2])
    #max_vel = np.array([np.inf, np.inf, np.inf])
    dmp.setConstraints(max_vel, max_acc)

    # Generate an output trajectory from the trained DMP
    dmp_p, dmp_dp, dmp_ddp, dmp_o, dmp_do, dmp_ddo, phase = dmp.rollout(tau, environment_scaling)

    scale = len(dmp_p_pre)/len(dmp_p)

    tp_pre = np.arange(0, len(dmp_p_pre) * 0.002, 0.002)

    tp = np.arange(0, len(dmp_p) * 0.002, 0.002)
    to = np.arange(0, len(dmp_o) * 0.002, 0.002)

    t_test = np.arange(0, len(dmp_o) * 0.002 * scale, scale * 0.002)
    print(t_test)

    #fig, ax = plt.subplots()
    #ax.plot(pre_phase, dmp_p_pre[:, 0])
    #ax.plot(post_phase, dmp_p[:,0])
    #ax.plot(post_phase, phase)
    #ax.set_xlim(ax.get_xlim()[::-1])

    #plt.show()

    #print(len(dmp_p_pre), len(dmp_p), len(tp), len(t_test))

    #dmp.plot(dmp_p_pre, dmp_p, tp_pre, t_test[0:-1], y_label=['X[m]', 'Y[m]', 'Z[m]'], title="Position of TCP", plot_demo=True)
    dmp.plot(dmp_dp_pre, dmp_dp, tp_pre, tp, y_label=['X[m/s]', 'Y[m/s]', 'Z[m/s]'], title="Velocity of TCP", plot_demo=True)
    #dmp.plot(dmp_ddp_pre, dmp_ddp, tp_pre, tp, y_label=['X[m/s^2]', 'Y[m/s^2]', 'Z[m/s^2]'], title="Acceleration of TCP", plot_demo=True)

    #dmp.plot3DDMP(dmp_p_pre,dmp_p, plot_demo=True)
    #dmp.plot(demo_axis, quaternion.as_rotation_vector(dmp_o), t, to, y_label=['X[rad]', 'Y[rad]', 'Z[rad]'], title="Orientation of TCP")





