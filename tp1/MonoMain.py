# -*- coding: utf-8 -*-
from Calibration import MonoCalibration
from Rectification import MonoRectification

#calbase = CalibrationBase()
# Acquisition
monocal = MonoCalibration(8)
#monocal.acquire()
# Calibration
rms, cameraMatrix, distCoeffs, imageSize = monocal.calibrate()
#monocal.visualizeBoards()

# Visualization
#monocal.plotRMS()

# Rectification
monorect = MonoRectification(cameraMatrix, distCoeffs, imageSize)
monorect.display()
