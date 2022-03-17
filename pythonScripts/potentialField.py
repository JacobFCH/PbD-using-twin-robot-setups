import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math

new_mesh = mesh.Mesh.from_file('/home/jacob/gits/super-duper-thesis/pythonScripts/DONUT_BOTTOM.stl')
mesh_normals = new_mesh.normals

mesh_v0 = new_mesh.v0
mesh_v1 = new_mesh.v1
mesh_v2 = new_mesh.v2

check_normals = np.cross(mesh_v1[0:3] - mesh_v0[0:3], mesh_v2[0:3] - mesh_v0[0:3])
#print(check_normals)
#print(mesh_normals[0:3])

# Normal for every v0 in the mesh 

#figure = pyplot.figure()
#axes = mplot3d.Axes3D(figure)

#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(new_mesh.vectors))

#scale = new_mesh.points.flatten()
#axes.auto_scale_xyz(scale, scale, scale)

#pyplot.show()

class potentialField():
    def __init__(self):
        self.x = 0
        self.o = 0
        self.d_ox = self.o - self.x

        self.gamma_o = 2500
        self.gamma_p = 2500
        self.gamma_d = 50
        self.k = 0.01
        self.beta2 = 20/np.pi

        mesh_list = np.array([])

    # Finds the element of the array that is nearest to the point specified as input
    def find_nearest(self, point_array, point):
        point_array = np.asarray(point_array)
        idx = (np.sum(np.abs(point_array - point),axis=1)).argmin()
        return idx

    # Computes the angle between the obstacle vector (od) and the force maginude (fm) vector
    def computeAngle(self, od, fm):
            angle = np.arccos(np.dot((od).T, fm) / (np.linalg.norm(od) * np.linalg.norm(fm)))
            return angle

    def projectForces(self, u, v, ang):
        return (-np.cos(ang) * (np.linalg.norm(u)) * (v/np.linalg.norm(v)))

    def logisticFunction(self, x):
        L = 1
        k = 4
        x0 = 1
        return 1 - (L/(1 + math.exp(-k*(x-x0))))

    def computePsi(self, od, fm, angle):
            r = np.cross((od), fm)
            r0 = r / np.linalg.norm(r)
            Rv = fm * np.cos(np.pi / 2) + np.cross(r0, fm * np.sin(np.pi / 2)) + np.dot(r0, fm) * r0 * (1 - np.cos(np.pi / 2))

            d = np.linalg.norm(od)
            psi = angle * np.exp(-self.beta2 * angle) * np.exp(-self.k * d)
            return psi, Rv

    def computeField(self, x, F, obs_nr):

        obs = np.linspace([4,4,0],[4,-4,0],9)
        norms = np.linspace([3,4,0],[3,-4,0],9) - obs
        obs_list = np.array([obs,obs,obs])

        #Compute effect from nearest point on the obstacle
        obs_idx = self.find_nearest(obs_list[obs_nr], x)
        obstacleVector_p = obs[obs_idx] - x
        print(obstacleVector_p)
        distance_vec = np.linalg.norm(obstacleVector_p)
        print("Distance",distance_vec)
        angle_p = self.computeAngle(norms[obs_idx], F)
        print("Angle",np.rad2deg(angle_p))
        projection = self.projectForces(F, norms[obs_idx], angle_p)
        print("Projction of F on the angle",projection)

        print( "Logi func", self.logisticFunction(distance_vec))
        squished_forces = np.array([0,0,0])
        if np.sum(projection) < 0:
            squished_forces = projection * self.logisticFunction(distance_vec)
        print( "Squished forces", squished_forces)

        force = F + squished_forces
        return force

field = potentialField()

np.set_printoptions(suppress=True)

obs_vec = np.asarray([3.5,1,0])

f_vec = np.asarray([2,2,0])

force = field.computeField(obs_vec, f_vec, 0)
print("Resulting force",force)