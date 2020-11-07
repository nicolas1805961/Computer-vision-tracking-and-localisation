# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d

from Segmentation import Segmentation
from Registration import Registration


### Segmentation
# Select zone of interest
#pcd = o3d.io.read_point_cloud("data/OtherBoxes/box4.ply")
#o3d.visualization.draw_geometries_with_editing([pcd])

# Load in Segmentation object
seg_box = Segmentation("data/cropped_box.ply")


# Remove Outliers
seg_box.removeOutliers()
# Compute normals
seg_box.computeNormals(alignVector=[0, 1, 0])
#seg_box.normalsHistogram()
# Estimate the floor's normal
floorNormal = seg_box.estimateFloorNormal()
# ALign the floor with the horizontal plane
seg_box.alignFloor()
# Remove the floor
seg_box.removeFloor(2)
seg_box.removeOutliers()

seg_box_5 = Segmentation("data/cropped_box_5.ply")
seg_box_5.removeOutliers()
seg_box_5.computeNormals(alignVector=[0, 1, 0])
floorNormal = seg_box_5.estimateFloorNormal()
seg_box_5.alignFloor()
seg_box_5.removeFloor(3)
seg_box_5.removeOutliers()
### Registration
# Load and visualize the objects

#seg_box = o3d.io.read_point_cloud("data/segmentation_box.ply")
#seg_box_2 = o3d.io.read_point_cloud("data/segmentation_box_2.ply")
reg = Registration(seg_box_5.pointCloud, seg_box.pointCloud)
# Global registration
reg.processGlobal()

# ICP Registration
reg.processICP()
reg.display()

