# Using synthetic data for improving detection of cyclists and pedestrians in autonomous driving

Author:       Zuzana Kopčilová (xkopci02) <br />
Institution:  Brno University of Technology, Faculty of Information Technology <br />
Date:         05/2023 <br />

## Thesis abstract
This thesis deals with creating a synthetic dataset for autonomous driving and the possibility of using it to improve the results of vulnerable traffic participants’ detection. Existing
works in this area either do not disclose the dataset creation process or are unsuitable for
3D object detection. Specific steps to create a synthetic dataset are proposed in this work,
and the obtained samples are validated by visualization. In the experiments, the samples
are then used to train the object detection model VoxelNet.

## Contents guide

```bash
./
├── dataset                 # Synthetic dataset samples in KITTI format
│   ├── calib               # Sensor calibration files (.txt)
│   ├── image_2             # RGB camera images (.png)
│   ├── label_2             # Label files (.txt)
│   └── velodyne            # Lidar point clouds (.bin)
├── scripts                 # Scripts for generating a synthetic dataset
├── text_src                # Technical report source code (LaTeX)
├── poster.pdf              # Poster presenting the thesis
└── technical_report.pdf    # Technical report
```
For more information about individual scripts of this solution and the environment in which they were used, please refer to the second README file that is included in the scripts folder.