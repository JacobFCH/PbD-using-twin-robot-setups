from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
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
        upsampledMesh = []

        if sampleMultiplier != 1:
            for i in range(sampleMultiplier-1):
                for j, triangle in enumerate(self.stl_mesh.vectors):
                    longestVertex = np.argmax([np.linalg.norm(triangle[1]-triangle[0]), np.linalg.norm(triangle[1]-triangle[2]), np.linalg.norm(triangle[0]-triangle[2])])
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

            return np.asarray(upsampledMesh)
        return self.stl_mesh.vectors

    def plotMesh(self):
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.stl_mesh.vectors))
        scale = self.stl_mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        pyplot.show()


objectList = np.array(["SCube"])
objectPose = np.array([[1.,0.,0.,0.12499999],
                       [0.,1.,0.,0.52499998],
                       [0.,0.,1.,0.50000006],
                       [0.,0.,0.,1.        ]])
objectMesh = STLMesh(objectList[0], objectPose, 1, 2)
objectMesh.plotMesh()