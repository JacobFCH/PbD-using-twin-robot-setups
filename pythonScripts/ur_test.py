import numpy as np
import time
from admittanceController import AdmittanceController
from scipy.spatial.transform.rotation import Rotation as R
import scipy
import rtde_receive
import rtde_io
import rtde_control

if __name__ == '__main__':

    ip = "192.168.1.111"
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    rtde_io = rtde_io.RTDEIOInterface(ip)
    rtde_c = rtde_control.RTDEControlInterface(ip)
    print("connected to: ", ip)

    dt = 1/500
    controller = AdmittanceController(dt, False)

    inital_pose = rtde_r.getActualTCPPose()

    velocity = 0.5
    acceleration = 0.5
    lookaheadtime = 0.1
    gain = 600

    tool_open = True
    tool_closed = False

    position_only = inital_pose

    # Wait for start command, green button
    while True:
        if rtde_r.getActualDigitalInputBits() == 32:
            break

    # Zero ft sensor before use
    rtde_c.zeroFtSensor()

    while True:
        startTime = time.time()

        force_torque = rtde_r.getActualTCPForce()
        compliant_frame = controller.computeCompliance(inital_pose, force_torque)
        #rtde_c.servoL(np.array([compliant_frame[0:3], position_only[3:6]]).flatten(), velocity, acceleration, dt/2, lookaheadtime, gain)
        #rtde_c.servoL(np.array([position_only[0:3], compliant_frame[3:6]]).flatten(), velocity, acceleration, dt / 2,lookaheadtime, gain)
        rtde_c.servoL(compliant_frame, velocity, acceleration, dt / 2, lookaheadtime, gain)

        # Stop the robot if the red button is pressed
        if rtde_r.getActualDigitalInputBits() == 128:
            print("Stopping robot")
            break

        diff = time.time() - startTime
        if diff < dt:
            time.sleep(dt - diff)

    rtde_c.servoStop()
