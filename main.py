import imp
import coppeliaSim.sim as sim # Import for simulation environment
from pythonScripts.admittanceController import AdmittanceController
import numpy as np
import time

class SimController():
    def __init__(self, ClientID, RobotName):
        self.simClientID = ClientID
        self.RobotName = RobotName

        self.jointHandles = np.array([-1, -1, -1, -1, -1, -1])
        self.jointHandleReturnCodes = np.array([-1, -1, -1, -1, -1, -1])
        self.tableHandleReturnCode, self.tableHandle = sim.simxGetObjectHandle(self.simClientID, "customizableTable",sim.simx_opmode_blocking)
        #self.sensor = FTsensor.FTsensor(ClientID)
        for i in range(6):
            self.jointHandleReturnCodes[i], self.jointHandles[i] = sim.simxGetObjectHandle(self.simClientID, self.RobotName + "_joint" + str(i + 1),sim.simx_opmode_blocking)

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
            self.curConfReturnCodes[i], self.curConf[i] = sim.simxGetJointPosition(self.simClientID, joint,
                                                                                   sim.simx_opmode_buffer)
        return self.curConf

    def getCurPose(self):
        ret, tip_handle = sim.simxGetObjectHandle(self.simClientID, self.RobotName + "_connection",sim.simx_opmode_blocking)
        pos = sim.simxGetObjectPosition(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        rot = sim.simxGetObjectOrientation(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        return pos[1] + rot[1]

    def getIKpath(self, pose, n_points=100):
        _, _, path, _, _ = sim.simxCallScriptFunction(self.simClientID, self.RobotName, sim.sim_scripttype_childscript,'ikPath',[n_points], pose,[], bytearray(), sim.simx_opmode_blocking)
        #print(path)
        return [path[x:x + 6] for x in range(0, len(path), 6)]

    def getNpoints(self, pose):
        curPose = self.getCurPose()

        curP = curPose[0:3]
        curR = curPose[3:6]
        goalP = pose[0:3]
        goalR = pose[3:6]
        distP = np.linalg.norm(np.asarray(goalP)-np.asarray(curP))
        distR = np.linalg.norm(np.asarray(goalR)-np.asarray(curR))

        Pppu = 10
        Rppu = 10

        return int(np.maximum(distP * Pppu, distR * Rppu))

    def moveJ(self, Q, max_vel, max_acc, max_jerk):
        inputInts = [max_vel, max_acc, max_jerk]
        sim.simxCallScriptFunction(self.simClientID, self.RobotName, sim.sim_scripttype_childscript, 'moveToConfig',inputInts, Q,[], bytearray(), sim.simx_opmode_blocking)
        complete = False
        cur_conf = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        i = 0
        t1 = time.time()
        while not complete:

            cur_conf = self.getCurConf()

            if np.abs(np.sum(Q - cur_conf)) < 0.01:
                complete = True
            #time.sleep(0.01)
            #if time.time() - t1 > 0.1:
            #    complete = True

        self.curConf = cur_conf
        return False

    def moveJPath(self, path, max_vel, max_acc, max_jerk):
        for i, Q in enumerate(path):
            done = self.moveJ(Q, max_vel, max_acc, max_jerk)
            if done:
                return True
        return True

    def moveL(self, pose, max_vel, max_acc, max_jerk):
        n_points = self.getNpoints(pose)
        path = self.getIKpath(pose,n_points)
        self.moveJPath(path, max_vel, max_acc, max_jerk)

if __name__ == "__main__":
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

    if clientID != -1:
        print('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res,objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print ('Number of objects in the scene: ',len(objs))
    else:
        print ('Remote API function call returned with error code: ',res)

    time.sleep(0.5)
    
    dt = 1/50
    simController = SimController(clientID, "UR10")
    controller = AdmittanceController(dt)

    desired_frame = [0.0, 0.5, 0.5, np.pi, 0, 0]
    force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Move to inital pose for testing
    simController.moveL(desired_frame, 10, 10, 10)

    timestep = 0
    print("Starting Test Loop")
    while timestep < 10:
        compliant_frame = controller.computeCompliance(desired_frame, force_torque)
        print(compliant_frame)
        simController.moveL(compliant_frame, 10, 10, 10)

        # Adding an external force a 1 second
        if timestep > 1 and timestep < 1 + dt:
            print("adding external force")
            force_torque = np.array([0.0,0.0,0.0,0.0,10.0,0.0])

        # Removing the external force at 4 seconds
        if timestep > 3 and timestep < 3 + dt:
            print("no external force")
            force_torque = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        timestep += dt

        time.sleep(dt)
    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID,'Hello CoppeliaSim!',sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
