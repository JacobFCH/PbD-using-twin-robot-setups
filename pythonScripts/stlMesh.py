from stl import mesh
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import numpy as np
import math

# Use example, creates an object of the class stl_mesh containing all the information about the mesh
# stl_mesh = STLMesh('DONUT_BOTTOM.stl')

# Uses numpy-stl library

class STLMesh():
    def __init__(self, file_name, T=np.eye(4), scale=1, upsampleFactor=1):
        self.stl_mesh = mesh.Mesh.from_file('C:/Users/jacob/OneDrive/Dokumenter/gits/super-duper-thesis/stlfiles/' + file_name + ".stl")

        # Scale and translate the mesh such that it fits into the scene in relation to the robot
        self.stl_mesh.vectors = self.stl_mesh.vectors * scale
        self.stl_mesh.transform(T)

        _, self.cog, _ = self.stl_mesh.get_mass_properties()

        # uVectors is the upsampled vectors given by the upsampleFactor, set to one by default
        self.uVectors = self.upsampleMesh(upsampleFactor)

        self.points = []
        self.v0 = []
        self.v1 = []
        self.v2 = []

        for vectors in self.uVectors:
            self.v0.append(vectors[0])
            self.v1.append(vectors[1])
            self.v2.append(vectors[2])

            self.points.append(vectors.flatten())

        self.points = np.asarray(self.points)
        self.v0 = np.asarray(self.v0)
        self.v1 = np.asarray(self.v1)
        self.v2 = np.asarray(self.v2)

        # Compute normals
        self.normals = np.cross(self.v1 - self.v0, self.v2 - self.v0)

    def upsampleMesh(self, sampleMultiplier):
        baseMesh = self.stl_mesh.vectors

        if sampleMultiplier != 1:
            for i in range(sampleMultiplier-1):
                upsampledMesh = []
                for j, triangle in enumerate(baseMesh):
                    longestVertex = np.argmax([np.linalg.norm(triangle[1]-triangle[0]), np.linalg.norm(triangle[1]
                                               - triangle[2]), np.linalg.norm(triangle[0]-triangle[2])])
                    if longestVertex == 0:
                        splitPoint = np.linspace(triangle[0], triangle[1], 3)[1]
                        upsampledMesh.append([triangle[0], splitPoint, triangle[2]])
                        upsampledMesh.append([splitPoint, triangle[1], triangle[2]])
                    elif longestVertex == 1:
                        splitPoint = np.linspace(triangle[1], triangle[2], 3)[1]
                        upsampledMesh.append([triangle[0], triangle[1], splitPoint])
                        upsampledMesh.append([triangle[0], splitPoint, triangle[2]])
                    elif longestVertex == 2:
                        splitPoint = np.linspace(triangle[2], triangle[0], 3)[1]
                        upsampledMesh.append([triangle[0], triangle[1], splitPoint])
                        upsampledMesh.append([splitPoint, triangle[1], triangle[2]])
                baseMesh = upsampledMesh
            return np.asarray(upsampledMesh)
        return self.stl_mesh.vectors

    def plotMesh(self):
        ax = a3.Axes3D(pl.figure())
        for i in range(len(self.uVectors)):
            tri = a3.art3d.Poly3DCollection([self.uVectors[i]])
            tri.set_color("white")
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)
        scale = self.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        pl.show()


#objectList = np.array(["SCube"])
#objectPose = np.array([[1.,0.,0.,0.12499999],
#                       [0.,1.,0.,0.52499998],
#                       [0.,0.,1.,0.50000006],
#                       [0.,0.,0.,1.        ]])
#objectMesh = STLMesh(objectList[0], objectPose, 1, 6)
#objectMesh.plotMesh()