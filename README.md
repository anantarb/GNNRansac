# GNN-Ransac

This repository shows how the DSAC* can be improved by training GNN to learn the confidences of 2D-3D correspondences. For GNNLine, follow the readme there.

# 1 Repository Organization
 * `models/`: It contains GNN and DSAC* network code.
 * `utils/`: It contains hypothesis utils for pose estimation problem, for data and for GNN network training.
 * `main.py`: It contains algorithm logic.

# 2 Setting up
 * To download the dataset, run `setup_7scenes.py` inside `dataset` folder.
 * You have to download the pre-trained scene coordinate network of dsac* from https://github.com/vislearn/dsacstar and put everything inside `saved_model/`.
 * The code is tested on `python 3.9`.
 * Install the requirements from `requirements.txt`.

# 3 Running the code
 * To train the network, run `python main.py`. The outputs are the median translation and rotation error for a specific scene based on what you pass as an argument.

# 4 Changing hyperparameters
 * You can also change hyperparameters. See arguments of `main.py` for this.

# 5 Algorithm logic
Our idea is a combination of https://arxiv.org/pdf/1905.04132.pdf, https://arxiv.org/pdf/1706.00984.pdf, https://ieeexplore.ieee.org/document/9394752 and https://arxiv.org/pdf/2103.09435.pdf. Here we train the GNN in a self-supervised manner to maximize the expected inlier score. We thank all the authors of these papers for their great work.

# Improvement ideas

We always welcome new ideas. There are many improvements that can be done here. One could be incorporating advanced concepts of reinforcement learning. Therefore, if you want to contribute make a pull request. 
 
 