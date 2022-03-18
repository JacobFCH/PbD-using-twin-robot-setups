from stl import mesh

# Use example, creates an object of the class stl_mesh containing all the information about the mesh
# stl_mesh = STLMesh('DONUT_BOTTOM.stl')

# Uses numpy-stl library

class STLMesh():
    def __init__(self, file_name):
        self.stl_mesh = mesh.Mesh.from_file('/home/jacob/gits/super-duper-thesis/pythonScripts/' + file_name)
        self.normals = self.stl_mesh.normals
        self.vertex0 = self.stl_mesh.v0
        self.vertex1 = self.stl_mesh.v1
        self.vertex2 = self.stl_mesh.v2

