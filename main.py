import coppeliaSim.sim as sim # Import for simulation environment
from pythonScripts.admittanceController import AdmittanceController
from pythonScripts.stlMesh import STLMesh
from pythonScripts.potentialField import potentialField
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
import time

class simController():

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

if __name__ == "__main__":
    np.set_printoptions(suppress=True) # Supress np scientific notation
    sim.simxFinish(-1)  # just in case, close all opened connections
    server_ip = '127.0.0.1'
    clientID = sim.simxStart(server_ip, 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

    if clientID != -1:
        print('Connected to remote API server with ip: ', server_ip)
        res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok: print('Remote API function call returned with error code: ', res)

        dt = 1/50
        UR5 = simController(clientID, "UR5")
        UR10 = simController(clientID, "UR10")
        controller = AdmittanceController(dt, False)

        desired_frame = [0.125, 0.225, 0.5, np.pi, 0.0, 0]
        compliant_frame = desired_frame
        force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        force = force_torque[0:3]

        objectPose = UR5.getObjectPose("SCube", "customizableTable")
        print(objectPose)

        objectList = np.array(["SCube"])
        objectMesh = STLMesh(objectList[0], objectPose, 1/1000)
        #objectMesh.plotMesh()

        field = potentialField(128,0.06)
        #field.plotLogiFunc()

        timestep = 10
        print("Starting Test Loop")
        while timestep < 6:
            startTime = time.time()

            force = field.computeField(compliant_frame[0:3], force, objectMesh.vertex0, objectMesh.normals)
            #print(force)
            force_torque[0:3] = force
            compliant_frame = controller.computeCompliance(desired_frame, force_torque)
            #print(compliant_frame)

            UR5.moveL(compliant_frame, 1, 1, 1)
            #UR10.moveL(compliant_frame, 1, 1, 1)

            # Adding an external force a 1 second
            if timestep > 0.3 and timestep < 0.32 + dt:
                print("adding external force")
                force = np.array([0,1,0])

            # Removing the external force at 4 seconds
            if timestep > 5 and timestep < 5 + dt:
                print("no external force")
                force = np.array([0.0,0.0,0.0])
            timestep += dt

            diff = time.time() - startTime
            if(diff < dt):
                time.sleep(dt-diff)

        sim.simxGetPingTime(clientID)
        sim.simxFinish(clientID)
    else:
        print('Failed connecting to remote API server')
    print('Program ended')