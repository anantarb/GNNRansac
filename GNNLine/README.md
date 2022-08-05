# GNN-Line

This repository shows how the DSAC Line code can be extended to make GNN learn give higher weight to inliers and lower weight to outliers.

# 1 Repository Organization
 * `gnn_dsac.py`: It contains DSAC utils to sample hypothesis using GNN weights or randomly. (Similar to `dsac.py` from DSACLine)
 * `line_dataset.py`: Same as `line_dataset.py` from DSACLine.
 * `line_loss.py`: Same as `line_loss.py` from DSACLine.
 * `line_nn.py`: Same as `line_nn.py` from DSACLine expect there is a GNN branch that takes in node features from middle of the DSACLine Network.
 * `main.py`: It contains code to train the networks. First it trains DSACLine Network to place most of the points on the line. Later, it trains GNN in a self-supervised manner to maximize expected score.
 * `visualization_helper.py`: Helper to visualize the results

# 2 Running the code
 * To train the network, run `python main.py`. We recommend to run this in a warning disabled mode to clearly visualize the expected score as the iteration progresses.

# 3 Ouputs
 * If you run `main.py`, the results will be saved inside `viz` folder. The results will have format visualization_{num}, where num is the iterations of training of GNN. The sampling weights should change as the iterations progresses, giving higher weight to inlier points and lower weights to outlier points.

# 4 Changing hyperparameters
 * You can also change hyperparameters. See arguments of `main.py` for this. 