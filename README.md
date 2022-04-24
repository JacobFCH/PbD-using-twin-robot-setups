# Learning from Demonstration using Twin Setups and Virtual Barriers

Master Thesis Project, Advanced Robotics Technology

The dependencies for the code can be found in the requirements.txt file and installed using:

```
pip install -r requirements.txt
```

The folder ```pythonScripts``` contains the files for the following parts:

- Admittance Controller
- Potential Fields
- 3D Mesh Loader
- Dynamic Movement Primitives
  - Demonstrations used by the DMPs

The folder ```stlfiles``` contains any .STL files that are used in the project

The folder ```coppeliaSim``` contains the files for the simulation tool CoppeliaSim

For this project CoppeliaSim 4.3 is used and can be found at https://www.coppeliarobotics.com/downloads

In order for the python remote api bindings to work the following flies files from the CoppeliaSim folder are used

```
programming/remoteApiBindings/python/python/sim.py
programming/remoteApiBindings/python/python/simConst.py
programming/remoteApiBindings/lib/lib/Windows/remoteApi.dll
```

If Linux is used the ```remoteApi.dll``` file needs to be replaced with the one corresponding to the OS used