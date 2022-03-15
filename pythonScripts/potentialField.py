import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

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
        return point_array[idx]

    # Computes the angle between the obstacle vector (od) and the force maginude (fm) vector
    def computeAngle(self, od, fm):
            angle = np.arccos(np.dot((od).T, fm) / (np.linalg.norm(od) * np.linalg.norm(fm)))
            return angle

    def computePsi(self, od, fm, angle):
            r = np.cross((od), fm)
            r0 = r / np.linalg.norm(r)
            Rv = fm * np.cos(np.pi / 2) + np.cross(r0, fm * np.sin(np.pi / 2)) + np.dot(r0, fm) * r0 * (1 - np.cos(np.pi / 2))

            d = np.linalg.norm(od)
            psi = angle * np.exp(-self.beta2 * angle) * np.exp(-self.k * d)
            return psi, Rv

    def computeField(self, x, F, obs_nr):

        p_o = np.zeros(3)
        p_p = np.zeros(3)
        p_d = np.zeros(3)

        obstacle_center = np.array([1,0,1])
        obs_list = np.array([[[1,1,1],[1,2,1],[0,1,1],[1,1,0],[1,0,1]],[[1,1,1],[1,2,1],[0,1,1],[1,1,0],[1,0,1]]])
        obstacle_radius = 0.5

        #Compute effect from obstacle center
        obstacle_vector_c = obstacle_center - x
        angle_o = self.computeAngle(obstacle_vector_c, F)
        psi_o, Rv_o = self.computePsi(obstacle_vector_c, F, angle_o)

        #Compute effect from nearest point on the obstacle
        obstacleVector_p = self.find_nearest(obs_list[obs_nr], x)
        angle_p = self.computeAngle(obstacleVector_p, F)
        psi_p, Rv_p = self.computePsi(obstacleVector_p, F, angle_p)

        Rv_avg = (Rv_o + Rv_p) * 0.5

        if not (angle_o > np.pi / 2 or angle_p > np.pi/2):
            p_o += self.gamma_o * Rv_o * psi_o
            p_p += self.gamma_p * Rv_p * psi_p
            p_d += self.gamma_d * Rv_avg * np.exp(-self.k*np.linalg.norm(obstacleVector_p))
        
        print(p_o,p_p,p_d)

        force = -F
        return force

field = potentialField()

np.set_printoptions(suppress=True)

obs_vec = np.asarray([1,0,0])

f_vec = np.asarray([0,1,0])

force = field.computeField(obs_vec, f_vec, 0)

print(field.computeAngle(obs_vec,f_vec))