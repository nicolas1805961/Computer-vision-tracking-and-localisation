#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

class PerspectiveCorrection:
    def __init__(self):
        self.windowName = 'Perspective Correction'

    def mouseCallback(self, event, x, y, flags, userData) : 
        # Implement the callback here
        if event == cv.EVENT_LBUTTONDOWN:
            userData['image'] = cv.circle(userData['image'],(x,y),10,(0,0,255),-1)
            cv.imshow(self.windowName, userData['image'])
        elif event == cv.EVENT_LBUTTONUP:
            userData['points'] = np.append(userData['points'], np.array([x, y], dtype=np.float32).reshape(1, 2), axis=0)

    def getCorners(self, image):
        userData = {}
        userData['image'] = image.copy()
        userData['points'] = np.empty((0, 2), dtype=np.float32)
        
        cv.imshow(self.windowName, image)
        
        while len(userData['points']) < 4:
            cv.setMouseCallback(self.windowName, self.mouseCallback, userData)
            cv.waitKey(1)
        
        return userData['points']

    def process(self, image, outSize):
        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL)
    
        targetPoints = np.array([[0, 0], [outSize[0], 0], [outSize[0], outSize[1]], [0, outSize[1]]], dtype=np.float32)
        
        print ('Click on the four corners of the object (top left -> top right -> bottom right -> bottom left)')
        sourcePoints = self.getCorners(image);
        
        # Compute the perspective transformation here
        M = cv.getPerspectiveTransform(sourcePoints, targetPoints)
        outImage = np.zeros(outSize, np.uint8)
        # Warp the source image to correct its perspective here
        outImage = cv.warpPerspective(image, M=M, dsize=outSize)
        
        while True:
            cv.imshow(self.windowName, outImage)
            key = cv.waitKey(0)
                
            if key == ord('\x1b') or key == ord('q'):
                break
            
        cv.destroyAllWindows()
        
        return outImage