import numpy as np
import math

class potentialField():
    def __init__(self, k, x0):
        self.L = 1  # Max value of the field, kept at one to have a value between 0 and 1
        self.k = k  # Field steepness, how fast does the field change
        self.x0 = x0  # Sigmoid midpoint

    # Finds the element of the array that is nearest to the point specified as input
    def find_nearest(self, point_array, point):
        point_array = np.asarray(point_array)
        idx = (np.sum(np.abs(point_array - point),axis=1)).argmin()
        return idx

    # Computes the angle between the obstacle vector (od) and the force maginude (fm) vector
    def computeAngle(self, od, fm):
            angle = np.arccos(np.dot((od), fm) / (np.linalg.norm(od) * np.linalg.norm(fm)))
            return angle

    # computes the scalar projection of the force towards the obstacle returning the component of the vector that is in the direction of the obstacle
    def projectForces(self, u, v):
        return (np.dot(u,v)/np.dot(v,v)) * v

    def logisticFunction(self, x):
        return 1 - (self.L/(1 + math.exp(-self.k*(x-self.x0))))

    def plotLogiFunc(self):
        import matplotlib.pyplot as plt
        x = np.linspace(-1, 1, 100)
        p = []
        for point in x:
            p.append(self.logisticFunction(point))
        plt.xlabel("x") 
        plt.ylabel("Logi(x)")  
        plt.plot(x, p) 
        plt.show()

    # Computes the repealing forces of the potential field and adds that to the input force
    def computeField(self, position, force, obstacle, obstacle_normals):

        idx = self.find_nearest(obstacle, position)
        obstacle_vector = np.asarray(obstacle[idx] - position)
        norm = np.asarray(obstacle_normals[idx])
        force = np.asarray(force)
        angle = self.computeAngle(norm, force)

        projection = self.projectForces(force, norm)

        if np.deg2rad(90) < angle < np.deg2rad(270):
            squished_forces = projection * self.logisticFunction(np.linalg.norm(obstacle_vector))
        else:
            squished_forces = np.array([0, 0, 0])

        return force - squished_forces
