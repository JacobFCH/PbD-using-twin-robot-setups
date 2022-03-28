from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np
import math

# Use example, creates an object of the class stl_mesh containing all the information about the mesh
# stl_mesh = STLMesh('DONUT_BOTTOM.stl')

# Uses numpy-stl library

class STLMesh():
    def __init__(self, file_name, T, scale):
        self.stl_mesh = mesh.Mesh.from_file('/home/jacob/gits/super-duper-thesis/stlfiles/' + file_name + ".stl")

        self.stl_mesh.transform(T)

        self.stl_mesh.vectors = self.stl_mesh.vectors * scale

        self.vertex0 = self.stl_mesh.v0
        self.vertex1 = self.stl_mesh.v1
        self.vertex2 = self.stl_mesh.v2
        self.normals = np.cross(self.vertex1 - self.vertex0, self.vertex2 - self.vertex0)

    def plotMesh(self):
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.stl_mesh.vectors))
        scale = self.stl_mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        pyplot.show()

    def area_or(self):
        normals = np.cross(self.vertex1 - self.vertex0, self.vertex2 - self.vertex0)
        normals_sum = np.array([math.fsum(normals[:, 0]),
                                   math.fsum(normals[:, 1]),
                                   math.fsum(normals[:, 2])])
        return normals_sum