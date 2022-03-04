import imp
from re import T
import coppeliaSim.sim as sim # Import for simulation environment
from pythonScripts.admittanceController import AdmittanceController
from pythonScripts.ikSolver import ikSolver
from scipy.spatial.transform.rotation import Rotation as R
from spatialmath import *
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
            self.curConfReturnCodes[i], self.curConf[i] = sim.simxGetJointPosition(self.simClientID, joint, sim.simx_opmode_buffer)
        return self.curConf

    def setNewConf(self, new_q):
        #deg_q = np.rad2deg(new_q)
        for i, joint in enumerate(self.jointHandles):
            self.curConfReturnCodes[i] = sim.simxSetJointTargetPosition(self.simClientID, joint, new_q[i], sim.simx_opmode_oneshot)

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

    desired_frame = [-0.11, 0.32, 0.52, 0.0, 0.0, 0.0]
    force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.set_printoptions(suppress=True)
    
    #start_conf = np.asarray([0.1, -0.3,  1.57081544,  0.34906918, -1.57079661, -0.00000048])
    start_conf = np.asarray([-1.64461192, -1.44273737,  1.87308305,  1.14045065, -1.57079633,  0.07381559])
    simController.setNewConf(start_conf)
    cur_q = start_conf
    print("After first move",cur_q)
    
    time.sleep(100)

    timestep = 0
    print("Starting Test Loop")
    while timestep < 2:
        startTime = time.time()
        compliant_frame = controller.computeCompliance(desired_frame, force_torque)
        T = simController.f2t(compliant_frame)

        q = ik.solveIK(T,cur_q)
        cur_q = simController.getCurConf()
        print(q)

        #print(q)
        simController.setNewConf(np.asarray(q))

        # Adding an external force a 1 second
        #if timestep > 1 and timestep < 1 + dt:
        #    print("adding external force")
        #    force_torque = np.array([0.01,0.0,0.0,0.0,0.0,0.0])

        # Removing the external force at 4 seconds
        #if timestep > 3 and timestep < 3 + dt:
        #    print("no external force")
        #    force_torque = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        timestep += dt

        diff = time.time() - startTime
        if(diff < dt):
            time.sleep(dt-diff)

    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID,'Hello CoppeliaSim!',sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
