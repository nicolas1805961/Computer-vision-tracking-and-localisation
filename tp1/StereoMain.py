# -*- coding: utf-8 -*-
from Calibration import StereoCalibration
from Rectification import StereoRectification
import cv2 as cv
import numpy as np

stereoCal = StereoCalibration(7, 6)
# Calibration
rms, cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, imageSizeLeft, R, T = stereoCal.calibrate()
# Visualization
#stereoCal.plotRMS()
# Rectification
stereoRect = StereoRectification(cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, imageSizeLeft, R, T)
stereoRect.computeCorrectionMaps()
leftFrame = cv.imread('data/stereo/MinnieRawLeft.png', cv.IMREAD_GRAYSCALE)
rightFrame = cv.imread('data/stereo/MinnieRawRight.png', cv.IMREAD_GRAYSCALE)
left, right = stereoRect.rectify(leftFrame, rightFrame)
#stereoRect.display(left, right)
# 3D reconstruction
#left = cv.imread('data/stereo/MinnieRawLeft.png', cv.IMREAD_GRAYSCALE)
#right = cv.imread('data/stereo/MinnieRawRight.png', cv.IMREAD_GRAYSCALE)
stereoRect.displayDisparity(left, right)
            
