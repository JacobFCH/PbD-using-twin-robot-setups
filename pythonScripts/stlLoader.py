from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Use example, creates an object of the class stl_mesh containing all the information about the mesh
# stl_mesh = STLMesh('DONUT_BOTTOM.stl')

# Uses numpy-stl library

class STLMesh():
    def __init__(self, file_name):
        self.stl_mesh = mesh.Mesh.from_file('/home/jacob/gits/super-duper-thesis/stlfiles/' + file_name + ".stl")
        self.normals = self.stl_mesh.normals
        self.vertex0 = self.stl_mesh.v0
        self.vertex1 = self.stl_mesh.v1
        self.vertex2 = self.stl_mesh.v2
    
    def plotMesh(self):
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.stl_mesh.vectors))
        scale = self.stl_mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        pyplot.show()
