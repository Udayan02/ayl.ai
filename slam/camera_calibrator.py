import cv2
import numpy as np
import glob

# Define the dimensions of checkerboard
CHECKERBOARD = (6, 9)  # Number of interior corners

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Extract frames from video
video = cv2.VideoCapture('store_vid1.mp4')
frameCount = 0

print(f"Num Frames: {video.get(cv2.CAP_PROP_FRAME_COUNT)}")

while True:
    success, img = video.read()
    if not success or frameCount >= 4000:
        break
        
    # Use every 30th frame (or similar)
    if frameCount % 30 == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(frameCount)
        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            
    frameCount += 1

video.release()
cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print camera matrix
print("Camera Matrix:")
print(mtx)
print("\nDistortion Coefficients:")
print(dist)