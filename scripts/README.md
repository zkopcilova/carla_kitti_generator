# Scripts for generating a synthetic dataset
This folder contains Python scripts needed to generate a synthetic dataset in KITTI format from the CARLA autonomous driving simulator.

## Environment
This solution was tested and used on a local machine running Windows 10 system and the CARLA version used was the 0.9.13 release for Windows ([available here](https://github.com/carla-simulator/carla/releases)).
Scripts from this solution should be placed into the CARLA repository as illustrated:

```bash
CARLA_0.9.13
├── CARLAUE4
├── Co-Simulation
├── Engine
├── HDMaps
├── Plugins
└── PythonAPI
    ├── carla
    ├── examples
    │   └── PLACE SCRIPTS HERE
    └── util
```

## Dependencies
- NumPy - 1.21.6
- Pillow - 9.5.0
- matplotlib - 3.5.3 


## Individual scripts
- generator_actors.py – Simulator environment setup, spawning actors
```bash
python3 generator_actors.py --host      [Optional] # IP of the host server (default: 127.0.0.1)
                            --port      [Optional] # TCP port to listen to (default: 2000)
                            --tm-port   [Optional] # Port to communicate with traffic manager (default: 8000)
                            --walkers   [Optional] # Number of walkers to spawn (default: 300)
                            --vehicles  [Optional] # Number of vehicles to spawn (default: 15)
                            --cyclists  [Optional] # Number of cyclists to spawn (default: 25)
                            --town      [Optional] # Which map to generate - values 1-5 (default: 2)
```

- generator_bbox.py – Utility script for bounding box conversion and object filtering

- generator_kitti.py – Main dataset creation script
```bash
python3 generator_kitti.py  --host            [Optional] # IP of the host server (default: 127.0.0.1)
                            --port            [Optional] # TCP port to listen to (default: 2000)
                            --frames          [Optional] # Number of frames to record (default: 10)
                            --starting_frame  [Optional] # Frame number to begin with (default: 0)
```

- generator_labels.py – Label file class

- generator_utils.py – Utility methods and constants for other scripts

## Usage
1. Make sure that CARLA server is running
2. Run generator_actors.py script to prepare simulator environment
3. Run generator_kitti.py script to start dataset creation process