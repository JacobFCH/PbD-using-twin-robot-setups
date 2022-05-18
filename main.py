import coppeliaSim.sim as sim  # Import for simulation environment
from pythonScripts.admittanceController import AdmittanceController
from pythonScripts.stlMesh import STLMesh
from pythonScripts.potentialField import potentialField
from pythonScripts.trajectory_generator import TrajectoryGenerator
from scipy.spatial.transform.rotation import Rotation as R
import scipy
import numpy as np
import time

import rtde_receive
import rtde_io
import rtde_control

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SimController():

    def __init__(self, ClientID, RobotName):
        self.simClientID = ClientID
        self.RobotName = RobotName

        self.jointHandles = np.array([-1, -1, -1, -1, -1, -1])
        self.jointHandleReturnCodes = np.array([-1, -1, -1, -1, -1, -1])
        self.tableHandleReturnCode, self.tableHandle = sim.simxGetObjectHandle(self.simClientID, "customizableTable",sim.simx_opmode_blocking)
        #self.sensor = FTsensor.FTsensor(ClientID)
        for i in range(6):
            self.jointHandleReturnCodes[i], self.jointHandles[i] = sim.simxGetObjectHandle(self.simClientID, self.RobotName + "_joint" + str(i + 1), sim.simx_opmode_blocking)

        if not self.jointHandleReturnCodes.all() == 0:
            exit("Failed to obtain jointHandles")

        self.curConf = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        self.curConfReturnCodes = np.array([-1, -1, -1, -1, -1, -1])

        for i, joint in enumerate(self.jointHandles):
            self.curConfReturnCodes[i], self.curConf[i] = sim.simxGetJointPosition(self.simClientID, joint, sim.simx_opmode_streaming)

        if not (self.curConfReturnCodes.all() == 0 or self.curConfReturnCodes.all() == 1):
            exit("Failed to obtain jointPositions")

    def getCurConf(self):
        for i, joint in enumerate(self.jointHandles):
            self.curConfReturnCodes[i], self.curConf[i] = sim.simxGetJointPosition(self.simClientID, joint, sim.simx_opmode_buffer)
        return self.curConf

    def getCurPose(self):
        _, tip_handle = sim.simxGetObjectHandle(self.simClientID, self.RobotName + "_connection", sim.simx_opmode_blocking)
        pos = sim.simxGetObjectPosition(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        rot = sim.simxGetObjectOrientation(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        return pos[1] + rot[1]

    def moveJ(self, Q, max_vel, max_acc, max_jerk):
        inputInts = [max_vel, max_acc, max_jerk]
        sim.simxCallScriptFunction(self.simClientID, self.RobotName, sim.sim_scripttype_childscript, 'moveToConfig', inputInts, Q, [], bytearray(), sim.simx_opmode_blocking)
        complete = False
        cur_conf = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        i = 0
        t1 = time.time()
        while not complete:

            cur_conf = self.getCurConf()

            if np.abs(np.sum(Q - cur_conf)) < 0.01:
                complete = True
            time.sleep(0.01)
            if time.time() - t1 > 5.0:
                complete = True

        self.curConf = cur_conf
        return False

    def moveToPose(self, pose):
        Q = self.solveIK(pose)
        self.moveJ(Q[0], 100, 100, 100)

    def getIKpath(self, pose, n_points=100):
        _, _, path, _, _ = sim.simxCallScriptFunction(self.simClientID, self.RobotName, sim.sim_scripttype_childscript, 'ikPath', [n_points], pose, [], bytearray(), sim.simx_opmode_blocking)
        return [path[x:x + 6] for x in range(0, len(path), 6)]

    def moveL(self, pose, max_vel, max_acc, max_jerk, point_override=2):
        # Calculate number of points
        # sim.simxClearIntegerSignal(self.simClientID, "contact", sim.simx_opmode_blocking)
        n_points = point_override #self.getNpoints(pose)
        path = self.getIKpath(pose,n_points)
        self.moveJPath(path, max_vel, max_acc, max_jerk)

    def moveJPath(self, path, max_vel, max_acc, max_jerk):
        for i, Q in enumerate(path):
            done = self.moveJ(Q, max_vel, max_acc, max_jerk)
            if done:
                return True
        return True

    def getObjectHandle(self, objectName):
        _, objectHandle = sim.simxGetObjectHandle(self.simClientID, objectName, sim.simx_opmode_blocking)
        return objectHandle

    def getObjectPose(self, objectName, relativeName):
        objctHandle = UR5.getObjectHandle(objectName)
        relativeHandle = UR5.getObjectHandle(relativeName)

        _, objectPosition = sim.simxGetObjectPosition(self.simClientID, objctHandle, relativeHandle, sim.simx_opmode_blocking)
        _, objectOrientation = sim.simxGetObjectOrientation(self.simClientID ,objctHandle, relativeHandle, sim.simx_opmode_blocking)

        objectTransform = np.eye(4)
        euler = R.from_euler('zyx', objectOrientation)
        objectTransform[0:3,0:3] = euler.as_matrix()
        objectTransform[0:3,3] = objectPosition

        return objectTransform

    def setJointAngles(self, joint_angles):
        for i in range(len(self.jointHandles)):
            _ = sim.simxSetJointPosition(self.simClientID, self.jointHandles[i], joint_angles[i], sim.simx_opmode_oneshot)


if __name__ == "__main__":
    np.set_printoptions(suppress=True) # Supress np scientific notation
    sim.simxFinish(-1)  # just in case, close all opened connections
    server_ip = '127.0.0.1'
    clientID = sim.simxStart(server_ip, 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

    if clientID != -1:
        print('Connected to remote API server with ip: ', server_ip)
        res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok: print('Remote API function call returned with error code: ', res)

        ip = "192.168.1.111"
        rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        rtde_io = rtde_io.RTDEIOInterface(ip)
        rtde_c = rtde_control.RTDEControlInterface(ip)
        print("Connected to robot with ip: ", ip)

        dt = 1 / 500
        controller = AdmittanceController(dt)

        initial_pose = rtde_r.getActualTCPPose()

        velocity = 0.5
        acceleration = 0.5
        lookaheadtime = 0.1
        gain = 600

        UR5 = SimController(clientID, "UR5")

        #objectPose = np.array([[1., 0., 0., -0.50],
        #                       [0., 1., 0., 0.002],
        #                       [0., 0., 1., 0.50000006],
        #                       [0., 0., 0., 1.]])
        objectPose = UR5.getObjectPose("SCube", "UR5_Base")
        objectMesh = STLMesh("SCube", objectPose, 0.005, 10)
        field = potentialField(128)

        # Transform the forces from the sensor frame to the tool frame at the tip of the gripper
        tool_transform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.170],
            [0, 0, 0, 1]
        ])

        # Create the adjoint matrix based on
        # https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-2-twists-part-2-of-2/#department
        adjoint_matrix = np.zeros([6,6])
        adjoint_matrix[0:3,0:3] = tool_transform[0:3, 0:3]
        adjoint_matrix[3:6, 3:6] = tool_transform[0:3, 0:3]
        p_skew = np.array([[0, -tool_transform[2, 3], tool_transform[1, 3]],
                           [tool_transform[2, 3], 0, -tool_transform[0, 3]],
                           [-tool_transform[1, 3], tool_transform[0, 3], 0]])
        adjoint_matrix[3:6,0:3] = p_skew @ tool_transform[0:3,0:3]

        # Wait for start command, green button
        print("Robot Ready")
        while True:
            if rtde_r.getActualDigitalInputBits() == 32:
                # rtde_c.teachMode() # enable this for teach mode tests
                break

        # Zero ft sensor before use
        rtde_c.zeroFtSensor()

        current_q = np.asarray(rtde_r.getActualQ())
        print("Current Pose:", current_q)
        UR5.setJointAngles(current_q)

        recording = []
        is_recording = False
        start_rec_time = 0
        stop_rec_time = 0

        while True:
            startTime = time.time()

            if rtde_r.getActualDigitalInputBits() == 64:
                start_rec_time = time.time()
                is_recording = True
            if rtde_r.getActualDigitalInputBits() == 16:
                stop_rec_time = time.time()
                is_recording = False

            force_torque = rtde_r.getActualTCPForce()
            current_pose = rtde_r.getActualTCPPose()

            if is_recording:
                recording.append(current_pose)

            # Rotate the force in the tcp to the orientation of the tcp
            rot = R.from_rotvec(current_pose[3:6])
            rMatrix = rot.as_matrix()
            invMatrix = scipy.linalg.inv(rMatrix)
            force_tcp = invMatrix @ force_torque[0:3]
            torque_tcp = invMatrix @ force_torque[3:6]

            wrench_transform = adjoint_matrix.T @ np.array([torque_tcp, force_tcp]).flatten()
            ft_tcp = np.array([wrench_transform[3:6], wrench_transform[0:3]]).flatten()

            #post_field_force = field.computeFieldEffect(current_pose[0:3], ft_tcp, objectMesh.v0, objectMesh.normals)
            compliant_frame = controller.computeCompliance(initial_pose, ft_tcp, rMatrix)
            rtde_c.servoL(compliant_frame, velocity, acceleration, dt / 2, lookaheadtime, gain)

            current_q = np.asarray(rtde_r.getActualQ())
            UR5.setJointAngles(current_q)

            # Stop the robot if the red button is pressed
            if rtde_r.getActualDigitalInputBits() == 128 or rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
                print("Stopping robot")
                break

            diff = time.time() - startTime
            if diff < dt:
                time.sleep(dt - diff)

        # rtde_c.endTeachMode() # Enable this for teach mode tests
        rtde_c.servoStop()

        testTrajectory = TrajectoryGenerator(0.15, -0.01)
        recording = np.asarray(recording)

        print("Duration: ", stop_rec_time - start_rec_time)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(testTrajectory.testTrajectory[:,0], testTrajectory.testTrajectory[:,1], testTrajectory.testTrajectory[:,2])
        ax.plot(recording[:,0], recording[:,1], recording[:,2])
        plt.show()

        print("Computing Dynamical Time Warping")
        x_dtw = testTrajectory.dtw(recording[:, 0], testTrajectory.testTrajectory[:, 0])
        y_dtw = testTrajectory.dtw(recording[:, 1], testTrajectory.testTrajectory[:, 1])
        z_dtw = testTrajectory.dtw(recording[:, 2], testTrajectory.testTrajectory[:, 2])
        print(x_dtw, y_dtw, z_dtw)

        sim.simxGetPingTime(clientID)
        sim.simxFinish(clientID)
    else:
        print('Failed connecting to remote API server')