# Use conda to install pybullet with pre-built wheel
import pybullet as p
import pybullet_data as data
import random

plane_size = 10
grass_fraction = random.uniform(0.8, 1.0)
grid_size = random.uniform(0.5, 1.0)

# Establish connection to simulation environment
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance = 12, cameraYaw = 50, cameraPitch = -35, cameraTargetPosition = [0 ,0 ,0]) # Camera Position

# Hides the extra GUI features that are not used 
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  


# Loading the plane 
plane_id = p.loadURDF("plane.urdf")
p.changeVisualShape(plane_id, -1, rgbaColor=[1, 1, 1, 1])  

# Placing grass on a portion of the plane
num_cells = int(plane_size / grid_size)
for i in range(num_cells):
    for j in range(num_cells):
        if random.random() < grass_fraction:
            # Squares representing grass
            pos_x = i * grid_size - plane_size / 2 + grid_size / 2
            pos_y = j * grid_size - plane_size / 2 + grid_size / 2
            # Grass Patches
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[grid_size / 2, grid_size / 2, 0.01],
                rgbaColor=[0.13, 0.55, 0.13, 1] # Grass colour 
            )
            p.createMultiBody(baseMass = 0, baseVisualShapeIndex = visual_shape,
                              basePosition = [pos_x, pos_y, 0])
            

while True:
    p.setTimeStep(1./240.)
    p.setRealTimeSimulation(1)