import numpy as np
import time
from admittanceController import AdmittanceController
import rtde_receive
import rtde_io
import rtde_control
from pythonScripts.potentialField import potentialField

if __name__ == '__main__':

    ip = "192.168.1.111"
    rtde_r = rtde_receive.RTDEReceiveInterface(ip)
    rtde_io = rtde_io.RTDEIOInterface(ip)
    rtde_c = rtde_control.RTDEControlInterface(ip)
    print("Connected to: ", ip)

    dt = 1/500
    controller = AdmittanceController(dt)

    initial_pose = rtde_r.getActualTCPPose()

    velocity = 0.5
    acceleration = 0.5
    lookaheadtime = 0.1
    gain = 600

    field = potentialField(128, 0.06)
    #field.plotLogiFunc()

    # Wait for start command, green button
    while True:
        if rtde_r.getActualDigitalInputBits() == 32:
            break

    # Zero ft sensor before use
    rtde_c.zeroFtSensor()

    while True:
        startTime = time.time()

        force_torque = rtde_r.getActualTCPForce()
        current_pose = rtde_r.getActualTCPPose()
        # Insert Potential field computation
        compliant_frame = controller.computeCompliance(initial_pose, force_torque)
        rtde_c.servoL(compliant_frame, velocity, acceleration, dt / 2, lookaheadtime, gain)

        # Stop the robot if the red button is pressed
        if rtde_r.getActualDigitalInputBits() == 128:
            print("Stopping robot")
            break

        diff = time.time() - startTime
        if diff < dt:
            time.sleep(dt - diff)

    rtde_c.servoStop()
