#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2 as cv
import numpy as np
from math import sqrt

from Homography import PerspectiveCorrection
from Tracking import Tracker
from PoseEstimation import PoseEstimator

#height_px = 1080# size in pixels along Y
#width_px = 1920# size in pixels along X
#size = (width_px, height_px)

#image = cv.imread('data/box_nature.jpg') # Read the image of your object

#cor = PerspectiveCorrection()
#outImage = cor.process(image, size)

# Save your image 
#cv.imwrite('data/box_nature_transformed.jpg', outImage)

# Implement the calibration results of TP1 Perception3D
cameraMatrix = np.array([[ 765.87773035, 0, 297.19547024],
                         [0, 768.59189293, 243.84358289],
                         [0, 0, 1 ]], np.float32)
distCoeffs = np.array([-1.70424057e-01, 7.93496455e-01, 7.15783705e-04, -7.93544931e-03, -6.30725266e-01], np.float32)
#
#
img_object = cv.imread('data/box_nature_transformed.jpg', cv.IMREAD_GRAYSCALE)# Read the image obtained in the first part
img_object = cv.resize(img_object, (250, 200))# Resize your object image
#
tracker = Tracker(img_object)
tracker.display()
#
# Dimension of the object in the real world
#objectSize = (0.13, 0.10, 0.10)

# Estimate the pose and reproject on the image
#posEst = PoseEstimator(img_object, objectSize, cameraMatrix, distCoeffs)
#posEst.display(cameraMatrix, distCoeffs)
