import numpy as np
import time
from admittanceController import AdmittanceController
from potentialField import potentialField
from stlMesh import STLMesh
import rtde_receive
import rtde_io
import rtde_control

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

    objectPose = np.array([[1.,0.,0.,-0.50],
                           [0.,1.,0.,0.002],
                           [0.,0.,1.,0.50000006],
                           [0.,0.,0.,1.        ]])
    objectMesh = STLMesh("SCube", objectPose, 1 / 100, 8)
    field = potentialField(128, 0.06)

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
        post_field_force = field.computeField(current_pose[0:3], force_torque[0:3], objectMesh.v0, objectMesh.normals)
        post_field_ft = np.array([post_field_force, force_torque[3:6]]).flatten()
        compliant_frame = controller.computeCompliance(initial_pose, post_field_ft)
        rtde_c.servoL(compliant_frame, velocity, acceleration, dt / 2, lookaheadtime, gain)

        # Stop the robot if the red button is pressed
        if rtde_r.getActualDigitalInputBits() == 128 or rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
            print("Stopping robot")
            break

        diff = time.time() - startTime
        if diff < dt:
            time.sleep(dt - diff)

    rtde_c.servoStop()
