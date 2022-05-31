import numpy as np
import math
import matplotlib.pyplot as plt

class potentialField():
    def __init__(self, k):
        self.L = 1  # Max value of the field, kept at one to have a value between 0 and 1
        self.k = k  # Field steepness, how fast does the field change
        # self.x0 = x0  # Sigmoid midpoint

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

    def logisticFunction(self, x, x0):
        return 1 - (self.L/(1 + math.exp(-self.k*(x-x0))))

    def plotLogiFunc(self):
        x = np.linspace(-1, 1, 100)
        p = []
        for point in x:
            p.append(self.logisticFunction(point))
        plt.xlabel("x") 
        plt.ylabel("Logi(x)")  
        plt.plot(x, p) 
        plt.show()

    # Computes the repealing forces of the potential field and adds that to the input force
    def computeForce(self, position, force, nearest_point, nearest_normal, mid_point):
        obstacle_vector = np.asarray(nearest_point - position)
        norm = np.asarray(nearest_normal)
        force = np.asarray(force)
        angle = self.computeAngle(norm, force)

        projection = self.projectForces(force, norm)

        logicFuncs = np.array([self.logisticFunction(np.linalg.norm(obstacle_vector), mid_point),
                               self.logisticFunction(np.linalg.norm(obstacle_vector), mid_point),
                               self.logisticFunction(np.linalg.norm(obstacle_vector), mid_point)])

        if np.deg2rad(90) < angle < np.deg2rad(270):
            squished_forces = projection * logicFuncs
        else:
            squished_forces = np.array([0, 0, 0])

        return force - squished_forces

    def computeFieldEffect(self, position, ft, mesh_vertices, mesh_normals):

        idx = self.find_nearest(mesh_vertices, position)
        post_field_ft = np.zeros(6)
        post_field_ft[0:3] = self.computeForce(position, ft[0:3], mesh_vertices[idx], mesh_normals[idx], 0.06)
        post_field_ft[3:6] = ft[3:6]

        return post_field_ft

field = potentialField(128)

neatest_point = np.array([0.5, 0.5, 0])
test_force = np.array([1, 0, 0])
test_normal = np.array([-1,0,0])

grid = []
x = np.linspace([0.1,0.1,0],[1,0.1,0], 10)
for i in range(10):
    y = np.linspace(x[i], np.array([x[i,0],1,0]), 10)
    grid.append(y)

grid = np.asarray(grid)

# Creating plot
fig, ax = plt.subplots()
ax.plot(neatest_point[0], neatest_point[1], marker="o", markersize=10, markeredgecolor="tab:red", markerfacecolor="tab:red")
ax.quiver(neatest_point[0], neatest_point[1], test_normal[0], test_normal[1], scale = 11, color="tab:red")
for i in range(len(grid[0])):
    for j in range(len(grid[0])):

        point = grid[j,i]
        squished_force = field.computeForce(point,test_force,neatest_point, test_normal, 0.3)

        ax.quiver(point[0], point[1], squished_force[0], squished_force[1], scale = 11, color="tab:blue")
ax.set_ylabel('Y')
ax.set_xlabel('X')

# Show plot
plt.show()

