"""
Configuration parameters for MARVEL simulation and training.

This module defines key parameters for:
- Folder and path configurations
- Simulation settings
- Drone and sensor characteristics 
- Map representation
- Training hyperparameters
- Neural network architecture
- Computational resource settings

Key configurations include:
- Number of agents
- Sensor ranges
- Map resolution
- Episode and training parameters
- GPU and logging options
"""

import os as _os

FOLDER_NAME = 'scratch_20261022_224221'
LOAD_FOLDER_NAME = 'marvel'
model_path = f'model/{FOLDER_NAME}'
load_path = f'load_model/{LOAD_FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 8
LOAD_MODEL = False
SAVE_IMG_GAP = 64
NUM_EPISODE_BUFFER = 40

# Sim parameters
N_AGENTS = 4
# Sensor profile. 'marvel' keeps the original V1 values. 'compact' shrinks the
# ranges for the 80 m maps from tools/gen_hetero_maps.py; both sensors there
# have the same nominal sector area (~56.5 m^2), so the scene decides which
# one sees more. Select with SENSOR_PROFILE below or MARVEL_SENSOR_PROFILE in
# the environment (Ray workers inherit it from the driver's shell).
SENSOR_PROFILE = _os.environ.get('MARVEL_SENSOR_PROFILE', 'marvel')
_SENSOR_PROFILES = {
    'marvel': dict(capabilities=((180.0, 8.0), (180.0, 8.0), (70.0, 15.0), (70.0, 15.0)),
                   sensor_range=10, initial_scan_range=10.0, frontier_range=20.0),
    'compact': dict(capabilities=((180.0, 6.0), (180.0, 6.0), (45.0, 12.0), (45.0, 12.0)),
                    sensor_range=12, initial_scan_range=6.0, frontier_range=16.0),
}
if SENSOR_PROFILE not in _SENSOR_PROFILES:
    raise ValueError(f'Unknown MARVEL_SENSOR_PROFILE {SENSOR_PROFILE!r}; use one of {sorted(_SENSOR_PROFILES)}')
_PROFILE = _SENSOR_PROFILES[SENSOR_PROFILE]

# Physical sensors only; legacy graph/observation/reward parameters stay below.
EPISODE_SENSOR_CAPABILITIES = _PROFILE['capabilities']
ENV_SEED = 0
INITIAL_SCAN_FOV = 360.0
INITIAL_SCAN_RANGE = _PROFILE['initial_scan_range']  # meter, shared initialization prior; <= shortest range
USE_CONTINUOUS_SIM = True
NUM_SIM_STEPS = 6
VELOCITY = 1
YAW_RATE = 35 # in degrees

# Heading parameters
FOV = 120   # in degrees
V_FOV = 60
MOUNTING_ANGLE = 15 # downwards
NUM_ANGLES_BIN = 36
NUM_DISTANCE_BINS = 10
FRONTIER_REPRESENTATION_RANGE = _PROFILE['frontier_range']  # meter, ~1.3x longest range; independent of capability
NUM_HEADING_CANDIDATES = 3
DRONE_HEIGHT = 2

# map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
# Must equal the LONGEST physical range: it sets the graph refresh radius,
# UTILITY_RANGE and UPDATING_MAP_SIZE used by every agent.
SENSOR_RANGE = _PROFILE['sensor_range']  # meter
UTILITY_RANGE = 0.9 * SENSOR_RANGE
MIN_UTILITY = 1

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# V1 team reward = removed frontier cells / TEAM_FRONTIER_NORMALIZER - 0.5.
# 39 = 10 m * 3.14 // FRONTIER_CELL_SIZE from the original MARVEL. It is kept
# for both profiles; recalibrate it to the mean removal per macro step on the
# maps you train on (see README, "Heterogeneous-sensing maps").
TEAM_FRONTIER_NORMALIZER = 39.0

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 256
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 4

# network parameters
NODE_INPUT_DIM = 6  # Legacy PolicyNet/QNet input; do not change for the V1 Actor.
ACTOR_NODE_INPUT_DIM = 5
ACTOR_COORDINATE_SCALE = 40.0  # meter, shared node/team coordinate normalization
CRITIC_NODE_INPUT_DIM = 6  # V1 GT features; legacy QNet still uses NODE_INPUT_DIM.
CRITIC_COORDINATE_SCALE = 40.0
EMBEDDING_DIM = 128

# Graph parameters
NUM_NODE_NEIGHBORS = 5
K_SIZE = NUM_NODE_NEIGHBORS**2   # the number of neighboring nodes
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1

USE_WANDB = False
TRAIN_ALGO = 3
# 0: SAC, 1:MAAC , 2: Ground Truth 3: MAAC and Ground Truth
