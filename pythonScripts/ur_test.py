import numpy as np
import time
from admittanceController import AdmittanceController
from trajectory_generator import TrajectoryGenerator
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

    velocity = 0.50
    acceleration = 0.50
    lookaheadtime = 0.1
    gain = 600

    #objectPose = np.array([[1.,0.,0.,-0.50],
    #                       [0.,1.,0.,0.002],
    #                       [0.,0.,1.,0.50000006],
    #                       [0.,0.,0.,1.        ]])
    #objectMesh = STLMesh("SCube", objectPose, 0.005, 10)
    #field = potentialField(128, 5)

    testTrajectory = TrajectoryGenerator(0.15, 0.30)

    # Wait for start command, green button
    print("Robot Waiting")
    while True:
        if rtde_r.getActualDigitalInputBits() == 32:
            break

    rtde_c.moveL(testTrajectory.testTrajectory[0])

    counter = 0

    while True:
        startTime = time.time()

        if counter >= len(testTrajectory.testTrajectory):
            break

        rtde_c.servoL(testTrajectory.testTrajectory[counter], velocity, acceleration, dt / 2, lookaheadtime, gain)

        counter += 1

        diff = time.time() - startTime
        if diff < dt:
            time.sleep(dt - diff)

    rtde_c.servoStop()

    '''
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
        #post_field_force = field.computeFieldEffect(current_pose[0:3], force_torque, objectMesh.v0, objectMesh.normals)
        #compliant_frame = controller.computeCompliance(initial_pose, np.array([post_field_force[0:3], force_torque[3:6]]).flatten())
        #rtde_c.servoL(compliant_frame, velocity, acceleration, dt / 2, lookaheadtime, gain)

        # Stop the robot if the red button is pressed
        if rtde_r.getActualDigitalInputBits() == 128 or rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
            print("Stopping robot")
            break

        diff = time.time() - startTime
        if diff < dt:
            time.sleep(dt - diff)
            
    rtde_c.servoStop()
    '''