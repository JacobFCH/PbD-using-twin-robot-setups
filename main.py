from re import T
import coppeliaSim.sim as sim # Import for simulation environment
from pythonScripts.admittanceController import AdmittanceController
from pythonScripts.ikSolver import ikSolver
import roboticstoolbox as rtb
from scipy.spatial.transform.rotation import Rotation as R
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
        deg_q = np.rad2deg(new_q)
        sim.simxPauseCommunication(self.simClientID, True)
        for i, joint in enumerate(self.jointHandles):
            self.curConfReturnCodes[i] = sim.simxSetJointTargetPosition(self.simClientID, joint, new_q[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.simClientID, False)

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

    desired_frame = [-0.4389, -0.1091, -0.05148, 0.0, 0.0, np.pi/2]
    force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.set_printoptions(suppress=True)
    
    #Initialize robot pose
    start_conf = np.asarray([0.0, -0.3490658504,  1.5707963268,  0.3490658504, -1.5707963268,  0])
    simController.setNewConf(start_conf)
    cur_q = start_conf
    print(cur_q)

    compliant_frame = controller.computeCompliance(desired_frame, force_torque)
    print(compliant_frame)

    T = simController.f2t(compliant_frame)

    q = ik.solveIK(T,cur_q)

    #time.sleep(100)

    #pose = UR5.fkine(start_conf)

    #rotm = R.from_matrix(np.array([[0,-1,0],[1,0,0],[0,0,1]]))

    #print(pose, rotm.as_euler('xyz'))

    #time.sleep(100)

    waypoint1 = np.array([-0.4385,     -0.1091,     -0.05148,    0.,          0.,          1.57079633])
    waypoint2 = np.array([-0.1385,     -0.1091,     -0.05148,    0.,          0.,          1.57079633])

    path = np.linspace(waypoint1[0:3],waypoint2[0:3])

    UR5 = rtb.models.DH.UR5()

    for pose in path:
        T = np.eye(4)
        rot = R.from_euler('xyz', desired_frame[3:6])
        T[0:3,0:3] = rot.as_matrix()
        T[0:3,3] = pose

        print(T)

        q = ik.solveIK(T, cur_q)
        cur_q = q
        print(q)

        print(UR5.fkine(q))

        simController.setNewConf(np.asarray(q))

        time.sleep(1)
        print("\n")

    '''    
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
    '''

    # Now close the connection to CoppeliaSim:
    sim.simxGetPingTime(clientID)
    sim.simxFinish(clientID)
