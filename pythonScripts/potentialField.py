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
print(check_normals)
print(mesh_normals[0:3])

# Normal for every v0 in the mesh 

#figure = pyplot.figure()
#axes = mplot3d.Axes3D(figure)

#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(new_mesh.vectors))

#scale = new_mesh.points.flatten()
#axes.auto_scale_xyz(scale, scale, scale)

#pyplot.show()