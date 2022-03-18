import coppeliaSim.sim as sim # Import for simulation environment
from pythonScripts.admittanceController import AdmittanceController
from pythonScripts.ikSolver import ikSolver
import roboticstoolbox as rtb
from spatialmath import *
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
import time

class simController():

    def __init__(self, ClientID, RobotName):
        self.simClientID = ClientID
        self.RobotName = RobotName

        self.jointHandles = np.array([-1, -1, -1, -1, -1, -1])
        self.jointHandleReturnCodes = np.array([-1, -1, -1, -1, -1, -1])
        self.tableHandleReturnCode, self.tableHandle = sim.simxGetObjectHandle(self.simClientID, "customizableTable",
                                                                               sim.simx_opmode_blocking)
        #self.sensor = FTsensor.FTsensor(ClientID)
        for i in range(6):
            self.jointHandleReturnCodes[i], self.jointHandles[i] = sim.simxGetObjectHandle(self.simClientID,
                                                                                           self.RobotName + "_joint" + str(
                                                                                               i + 1),
                                                                                           sim.simx_opmode_blocking)

        if not self.jointHandleReturnCodes.all() == 0:
            exit("Failed to obtain jointHandles")

        self.curConf = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        self.curConfReturnCodes = np.array([-1, -1, -1, -1, -1, -1])

        for i, joint in enumerate(self.jointHandles):
            self.curConfReturnCodes[i], self.curConf[i] = sim.simxGetJointPosition(self.simClientID, joint,
                                                                                   sim.simx_opmode_streaming)

        if not (self.curConfReturnCodes.all() == 0 or self.curConfReturnCodes.all() == 1):
            exit("Failed to obtain jointPositions")

    def getCurConf(self):
        for i, joint in enumerate(self.jointHandles):
            self.curConfReturnCodes[i], self.curConf[i] = sim.simxGetJointPosition(self.simClientID, joint,
                                                                                   sim.simx_opmode_buffer)
        return self.curConf

    def getCurPose(self):
        ret, tip_handle = sim.simxGetObjectHandle(self.simClientID, self.RobotName + "_connection",
                                                  sim.simx_opmode_blocking)
        pos = sim.simxGetObjectPosition(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        rot = sim.simxGetObjectOrientation(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        return pos[1] + rot[1]

    def moveJ(self, Q, max_vel, max_acc, max_jerk, is_measuring):
        inputInts = [max_vel, max_acc, max_jerk]
        sim.simxCallScriptFunction(self.simClientID, self.RobotName, sim.sim_scripttype_childscript, 'moveToConfig',
                                   inputInts, Q,
                                   [], bytearray(), sim.simx_opmode_blocking)
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

            if is_measuring:
                F, _ = self.sensor.getreading()
                # print(F)
                contact = self.sensor.iscontact(F)
                if contact:
                    self.sensor.savesensorframe()
                    # print("Contact!\n")
                    return True

        self.curConf = cur_conf
        return False

    def moveToPose(self, pose):
        Q = self.solveIK(pose)
        self.moveJ(Q[0], 100, 100, 100)

    def getIKpath(self, pose, n_points=100):
        _, _, path, _, _ = sim.simxCallScriptFunction(self.simClientID, self.RobotName, sim.sim_scripttype_childscript,
                                                      'ikPath',
                                                      [n_points], pose,
                                                      [], bytearray(), sim.simx_opmode_blocking)
        return [path[x:x + 6] for x in range(0, len(path), 6)]

    def getNpoints(self, pose):
        curPose = self.getCurPose()

        curP = curPose[0:3]
        curR = curPose[3:6]
        goalP = pose[0:3]
        goalR = pose[3:6]
        distP = np.linalg.norm(np.asarray(goalP)-np.asarray(curP))
        distR = np.linalg.norm(np.asarray(curR)-np.asarray(curP))

        Pppu = 50
        Rppu = 10

        return int(np.maximum(distP * Pppu, distR * Rppu))

    def moveL(self, pose, max_vel, max_acc, max_jerk, is_measuring=False):
        # Calculate number of points
        # sim.simxClearIntegerSignal(self.simClientID, "contact", sim.simx_opmode_blocking)
        n_points = 2 #self.getNpoints(pose)
        path = self.getIKpath(pose,n_points)
        self.moveJPath(path, max_vel, max_acc, max_jerk, is_measuring)

    def moveJPath(self, path, max_vel, max_acc, max_jerk, is_measuring):
        for i, Q in enumerate(path):
            done = self.moveJ(Q, max_vel, max_acc, max_jerk, is_measuring)
            if done:
                return True
        return True

if __name__ == "__main__":
    print('Program started')
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

    if clientID != -1:
        print('Connected to remote API server')

        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print('Number of objects in the scene: ', len(objs))
        else:
            print('Remote API function call returned with error code: ', res)

        time.sleep(2)

        simCont = simController(clientID, "UR5")

        #pose = [0.125, 0.225, 0.5, np.pi, 0, 0]
        #simController.moveL(pose, 1, 1, 1, 0)

        dt = 1/50
        controller = AdmittanceController(dt, True)
        ik = ikSolver()

        desired_frame = [0.125, 0.225, 0.5, np.pi, 0.0, 0]
        force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        np.set_printoptions(suppress=True)

        timestep = 0
        print("Starting Test Loop")
        while timestep < 5:
            startTime = time.time()
            compliant_frame = controller.computeCompliance(desired_frame, force_torque)

            simCont.moveL(compliant_frame, 1, 1, 1)

            # Adding an external force a 1 second
            if timestep > 0.3 and timestep < 0.32 + dt:
                print("adding external force")
                force_torque = np.array([0.0,0.0,0.2,0.0,0.0,0.0])

            # Removing the external force at 4 seconds
            if timestep > 1.5 and timestep < 1.5 + dt:
                print("no external force")
                force_torque = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
            timestep += dt

            diff = time.time() - startTime
            if(diff < dt):
                time.sleep(dt-diff)

        # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive.
        # You can guarantee this with (for example):
        sim.simxGetPingTime(clientID)
        # Now close the connection to CoppeliaSim:
        sim.simxFinish(clientID)
    else:
        print('Failed connecting to remote API server')
    print('Program ended')

'''
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
            self.curConfReturnCodes[i], self.curConf[i] = sim.simxGetJointPosition(self.simClientID, joint, sim.simx_opmode_buffer)
        return self.curConf

    def setNewConf(self, q, joint):
        _ = sim.simxSetJointTargetPosition(self.simClientID, self.jointHandles[joint], q, sim.simx_opmode_oneshot)

        #sim.simxPauseCommunication(self.simClientID, False)
        #sim.simxPauseCommunication(self.simClientID, True)
        #for i, joint in enumerate(self.jointHandles):
        #    self.curConfReturnCodes[i] = sim.simxSetJointTargetPosition(self.simClientID, joint, new_q[i], sim.simx_opmode_oneshot)
        #sim.simxPauseCommunication(self.simClientID, False)

    def getCurPose(self):
        ret, tip_handle = sim.simxGetObjectHandle(self.simClientID, self.RobotName + "_connection",sim.simx_opmode_blocking)
        pos = sim.simxGetObjectPosition(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        rot = sim.simxGetObjectOrientation(self.simClientID, tip_handle, self.tableHandle, sim.simx_opmode_blocking)
        return pos[1] + rot[1]

    def f2t(self, frame):
        T = np.eye(4)
        r = R.from_euler('xyz',frame[3:6])
        T[0:3,0:3] = r.as_matrix()
        T[0:3,3] = frame[0:3]
        return T


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
    simController = SimController(clientID, "UR5")
    controller = AdmittanceController(dt)
    ik = ikSolver()

    desired_frame = [-0.4389, -0.1091, -0.05148, 0.0, 0.0, np.deg2rad(90)]
    force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.set_printoptions(suppress=True)
    
    #Initialize robot pose
    start_conf = np.asarray([0.0, -0.3490658504,  1.5707963268,  0.3490658504, -1.5707963268,  0])
    #simController.setNewConf(start_conf)
    cur_q = start_conf
    #print(cur_q)

    compliant_frame = controller.computeCompliance(desired_frame, force_torque)
    #print(compliant_frame)

    T = simController.f2t(compliant_frame)

    q = ik.solveIK(T,cur_q)

    #time.sleep(100)

    #pose = UR5.fkine(start_conf)

    #rotm = R.from_matrix(np.array([[0,-1,0],[1,0,0],[0,0,1]]))

    #print(pose, rotm.as_euler('xyz'))

    #time.sleep(100)

    waypoint1 = np.array([-0.4385, -0.1091, -0.05148])
    waypoint2 = np.array([-0.1385, -0.1091, -0.05148])

    path = np.linspace(waypoint1,waypoint2, 200)

    UR5 = rtb.models.DH.UR5()

    for pose in path:
        T = np.eye(4)
        rot = R.from_euler('xyz', desired_frame[3:6])
        T[0:3,0:3] = rot.as_matrix()
        T[0:3,3] = pose

        #print(T)

        q = ik.solveIK(T, cur_q)
        cur_q = q
        print(cur_q)

        sim.simxPauseCommunication(simController.simClientID, True)
        simController.setNewConf(q[0], 0)
        simController.setNewConf(q[1], 1)
        simController.setNewConf(q[2], 2)
        simController.setNewConf(q[3], 3)
        #simController.setNewConf(q[4], 4)
        #simController.setNewConf(q[5], 5)
        sim.simxPauseCommunication(simController.simClientID, False)
        #print(UR5.fkine(q))
        #sim.simxPauseCommunication(simController.simClientID, True)
        #for i, qs in enumerate(q):
        #    simController.setNewConf(qs, i)
        #sim.simxPauseCommunication(simController.simClientID, False)
        time.sleep(0.02)
        #print("\n")

  
    timestep = 0
    print("Starting Test Loop")
    while timestep < 10:
        startTime = time.time()
        compliant_frame = controller.computeCompliance(desired_frame, force_torque)
        #print(compliant_frame)
        T = simController.f2t(compliant_frame)

        q = ik.solveIK(T,cur_q)
        cur_q = q
        #cur_q = simController.getCurConf()
        #print(q)

        simController.setNewConf(np.asarray(q))

        # Adding an external force a 1 second
        if timestep > 1 and timestep < 1 + dt:
            print("adding external force")
            force_torque = np.array([1.0,0.0,0.0,0.0,0.0,0.0])

        # Removing the external force at 4 seconds
        if timestep > 3 and timestep < 3 + dt:
            print("no external force")
            force_torque = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        timestep += dt

        diff = time.time() - startTime
        if(diff < dt):
            time.sleep(dt-diff)


    # Now close the connection to CoppeliaSim:
    sim.simxGetPingTime(clientID)
    sim.simxFinish(clientID)
    '''