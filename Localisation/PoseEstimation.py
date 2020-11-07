#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from Tracking import Tracker

def nullFunction(x = None, y = None):
    pass

class PoseEstimator(Tracker):
    def __init__(self, referenceImage, objectSize, cameraMatrix, distCoeffs, detectorType = 'ORB'):
        """ Init
        
        :param referenceImage: Reference image to be tracked
        :param objectSize: Size of the real object, (length, width, depth)
        :param detectorType: Type of the detector (ORB or AKAZE)
        """
        Tracker.__init__(self, referenceImage, detectorType)
        
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        
        self.boxPoints = np.float32([(0, 0, 0), (0.13, 0, 0), (0.13, -0.10, 0), (0, -0.10, 0), (0, 0, -0.10), (0.13, 0, -0.10), (0.13, -0.10, -0.10), (0, -0.10, -0.10)])
        
        self.objectCorners = np.array([[self.boxPoints[0]],
                                  [self.boxPoints[1]],
                                  [self.boxPoints[2]],
                                  [self.boxPoints[3]]], dtype=np.float32)
                                       
    def computePose(self):
        rvecs = []
        tvecs = []
        
        if len(self.homography) > 0:
            # Compute corners with perspectiveTransform
            corners = cv.perspectiveTransform(self.targetCorners, self.homography)
            
            # Solve the PnP problem
            ret,rvecs, tvecs = cv.solvePnP(self.objectCorners, corners, self.cameraMatrix, self.distCoeffs)
        return rvecs, tvecs
    
    def drawObjectBox(self, image, rvecs, tvecs):
        if len(rvecs) == 3:
            # Implement projectPoints here         
            imagePoints, jac = cv.projectPoints(self.boxPoints, rvecs, tvecs, self.cameraMatrix, self.distCoeffs)
            imagePoints = np.int32(imagePoints).reshape(-1,2)
            image = cv.drawContours(image, [imagePoints[:4]],-1,(0,255,0),3)
            
            for i,j in zip(range(4),range(4,8)):
                image = cv.line(image, tuple(imagePoints[i]), tuple(imagePoints[j]),(255),3)
    
            image = cv.drawContours(image, [imagePoints[4:]],-1,(0,0,255),3)
        
        return image
    
    def display(self, cameraMatrix, distCoeffs, cameraId = 0):
        capture = cv.VideoCapture(cameraId)
        cv.namedWindow('Pose estimation', cv.WINDOW_NORMAL)
        
        # create trackbars
        cv.createTrackbar('Matching Ratio (%)', 'Pose estimation', 70, 100, nullFunction)
        cv.createTrackbar('Minimum Matches', 'Pose estimation', 12, 50, nullFunction)
        cv.createTrackbar('Accumulation Weigth (%)', 'Pose estimation', 40, 100, nullFunction)
        cv.createTrackbar('Maximum Frames Lost', 'Pose estimation', 5, 100, nullFunction)
        
        while(True):
            ret, image = capture.read()
            imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            
            matchingRatio = cv.getTrackbarPos('Matching Ratio (%)', 'Pose estimation') / 100
            minMatches = cv.getTrackbarPos('Minimum Matches', 'Pose estimation')
            weight = cv.getTrackbarPos('Accumulation Weigth (%)', 'Pose estimation') / 100
            maxframeLost = cv.getTrackbarPos('Maximum Frames Lost', 'Pose estimation')
            
            keypoints, matches = self.detectAndMatch(imageGray)
            goodPoints, targetPoints, observedPoints = self.getCorrespondencePoints(keypoints, matches, matchingRatio)
            self.computeHomography(targetPoints, observedPoints, minMatches, weight)
            
            rvecs, tvecs = self.computePose()
            image = self.drawObjectBox(image, rvecs, tvecs)
                            
            cv.imshow('Pose estimation', cv.flip(image, 1))
    
            key = cv.waitKey(1)
            if key == ord('\x1b') or key == ord('q'):
                break
        
        capture.release()
        cv.destroyAllWindows()

        