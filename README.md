# LiRaPoc:  Targetless Extrinsic Calibration of LiDAR and Radar with Spatial Consistency of Polar Occupancy
## Absract
## Run 
Owing to the capability for reliable and all-weather long-range sensing, the fusion of Radar and LiDAR has been
widely applied to autonomous vehicles for robust perception. In practical operation, well manually calibrated extrinsic parameters, which are crucial for the fusion of multi-modal sensors, may drift due to the vibration. To address this issue, we present
a novel targetless calibration approach, termed LiRaCo, for the extrinsic calibration of LiDAR and Radar sensors. Although
both types of sensors can obtain geometric information, bridging the geometric correspondences between multi-modal data
without any clues of explicit artificial markers is nontrivial,
mainly due to the low vertical resolution of Radar. To achieve
the targetless calibration, LiRaCo utilizes a spatial occupancy
consistency between LiDAR point clouds and Radar scans in
a common cylindrical representation, considering the working
principles of two sensors. Specifically, LiRaCo expands valid
Radar pixel into 3D occupancy grid to constrain LiDAR point
clouds based on spatial consistency. Consequently, a cost function
involving extrinsic calibration parameters is defined from the
spatial overlap of 3D grid and LiDAR points. Parameters are
finally estimated by optimizing the cost function. Comprehensive
quantitative and qualitative experiments on two real outdoor
datasets with different LiDAR sensors demonstrate the feasibility
and accuracy of the proposed method.
```
python lirapoc.py --dataset boreas --datapath MY_PATH/LiRaPoc/data/boreas/ --datasequence boreas_example --method LiRaPoc
```
method opt: 
--method LiRaPoc (ours )/ 
--method icp / 
--method mt (match template)
