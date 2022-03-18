import numpy as np
import math

class potentialField():
    def __init__(self, k, x0):
        self.L = 1 # Max value of the field, kept at one to have a value between 0 and 1
        self.k = k # Field steepness, how fast does the field change
        self.x0 = x0 # Sigmoi midtpoint

    # Finds the element of the array that is nearest to the point specified as input
    def find_nearest(self, point_array, point):
        point_array = np.asarray(point_array)
        idx = (np.sum(np.abs(point_array - point),axis=1)).argmin()
        return idx

    # Computes the angle between the obstacle vector (od) and the force maginude (fm) vector
    def computeAngle(self, od, fm):
            angle = np.arccos(np.dot((od).T, fm) / (np.linalg.norm(od) * np.linalg.norm(fm)))
            return angle

    # computes the scalar projection of the force towards the obstacle returning the component of the vector that is in the direction of the obstacle
    def projectForces(self, u, v, ang):
        return (-np.cos(ang) * (np.linalg.norm(u)) * (v/np.linalg.norm(v)))

    def logisticFunction(self, x):
        return 1 - (self.L/(1 + math.exp(-self.k*(x-self.x0))))

    # Computes the repeling forces of the potential field and adds that to the input force
    def computeField(self, position, force, obstacle, obstacle_normals):

        idx = self.find_nearest(obstacle, position)
        obstacle_vector = obstacle[idx] - position
        angle = self.computeAngle(obstacle_normals[idx], force)
        projection = self.projectForces(force, obstacle_normals[idx], angle)
        
        if angle > np.deg2rad(90) and angle < np.deg2rad(270):
            squished_forces = projection * self.logisticFunction(np.linalg.norm(obstacle_vector))
        else:
            squished_forces = np.array([0,0,0])

        return force + squished_forces


field = potentialField(4,1)

np.set_printoptions(suppress=True)

position = np.asarray([0,0,0])

force = np.asarray([0,0,0])

obs = np.linspace([4,4,0],[4,-4,0],9)
norms = np.linspace([3,4,0],[3,-4,0],9) - obs

from admittanceController import AdmittanceController
import time

dt = 1/50
controller = AdmittanceController(dt, False)

desired_frame = [0, 0, 0, 0, 0.0, 0]
compliant_frame = desired_frame
force_torque = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
force = force_torque[0:3]

timestep = 0
print("Starting Test Loop")
while timestep < 20:
    startTime = time.time()
    force = field.computeField(compliant_frame[0:3], force, obs, norms)
    force_torque[0:3] = force
    compliant_frame = controller.computeCompliance(desired_frame, force_torque)
    print(compliant_frame, force)

    # Adding an external force a 1 second
    if timestep > 1 and timestep < 1 + dt:
        print("adding external force")
        force = np.array([2,0.0,0.0])

    # Removing the external force at 4 seconds
    if timestep > 18 and timestep < 18 + dt:
        print("no external force")
        force = np.array([0.0,0.0,0.0])
    timestep += dt

    diff = time.time() - startTime
    if(diff < dt):
        time.sleep(dt-diff)