# Use conda to install pybullet with pre-built wheel
import pybullet as p
import pybullet_data as data
import random
import numpy as np

plane_size = 10
grass_fraction = random.uniform(0.95, 1.0)
grid_size = random.uniform(1.27, 1.3)
print(grid_size)
# Establish connection to simulation environment
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance = 10, cameraYaw = 90, cameraPitch = -30, cameraTargetPosition = [0 ,0 ,0]) # Camera Position

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
obtsacles = [(random.randint(0, num_cells), random.randint(0, num_cells)) for _ in range(random.randint(1, 3))]  # 1-3 radnom points where obstacles will appear

# Grass texture to place in sim and more textures for obstacles
grass_texture = p.loadTexture("Textures/grass.jpg")  
soil_texture = p.loadTexture("Textures/soil.jpg")
flowers_texture = p.loadTexture("Textures/flowers.jpg")

for i in range(num_cells):
    for j in range(num_cells):
        # Deafult flag
        texture = True
        # Squares representing grass
        pos_x = i * grid_size - plane_size / 2 + grid_size / 2
        pos_y = j * grid_size - plane_size / 2 + grid_size / 2

        # Obstacle or not
        distance = min(np.linalg.norm(np.array([i, j]) - np.array(c)) for c in obtsacles) # Computes distance from current position to nearest cluster

        if distance < 1.5:
            texture = False
            colour = [1, 1, 1, 1] # Obstacle
        else:
            colour = [1, 1, 1, 1] 

        # Grass Patches
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents = [grid_size / 2, grid_size / 2, 0.01],
            rgbaColor = colour  
        )
        # Id for each tile/grass block
        tile_id = p.createMultiBody(
            baseMass = 0,
            baseVisualShapeIndex = visual_shape,
            basePosition =[ pos_x, pos_y, 0]
        )
        
        if texture:
            p.changeVisualShape(
                tile_id, 
                -1, # Links of object, -1 is base
                textureUniqueId = grass_texture
            )
        else:
            p.changeVisualShape(
                tile_id, 
                -1, # Links of object, -1 is base
                textureUniqueId = random.choice([soil_texture, flowers_texture])
            )
        

# Loading the lawn mower
mowBot_id = p.loadURDF("mower.urdf", basePosition = [5, 3.5, 0])


while True:
    p.setTimeStep(1./240.)
    p.setRealTimeSimulation(1)