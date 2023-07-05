import cv2
from hog import *
import numpy as np
from skimage.feature import hog

def processImage(image):
    # Convert image to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Equalize the histogram
    image = cv2.equalizeHist(image)
    return image

def getHogFromLandmarks(landmarks, image, radius=10, pixels_per_cell=(10, 10), cells_per_block=(2, 2), orientations=8, useSkimage=True, preprocess=True):

    features = []

    if preprocess:
        image = processImage(image) # REMOVE AND DO ONCE

    for landmark in landmarks:
        # Extract the patch around the landmark
        x, y = landmark
        x= max(0,x) 
        y= max(0,y)
        x= min(x, image.shape[1]-1) 
        y= min(y, image.shape[0]-1)
        #Check if the patch is out of bounds
        if (x - radius < 0) or (x + radius >= image.shape[1]) or (y - radius < 0) or (y + radius >= image.shape[0]):
            #Add zero padding
            image = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)

            #update x,y
            x += radius
            y += radius
        
        patch = image[y - radius:y + radius, x - radius:x + radius]
        if useSkimage:
            fd = hog(patch, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                            cells_per_block=cells_per_block, visualize=False, multichannel=False)
        else:
            fd = hog.CalculateHog(patch, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        features.extend(fd)

    return np.array(features)

def getSiftFromLandmarks(landmarks, image, radius=10, preprocess=True):

    features = np.zeros((len(landmarks), 128))

    if preprocess:
        image = processImage(image) # REMOVE AND DO ONCE

    keypoints = []

    for i in range(len(landmarks)):
        # Extract the patch around the landmark
        x, y = landmarks[i]
        x= max(0,x) #TODO: REMOVE NEGATIVES FROM DATASET
        y= max(0,y)
        x= min(x, image.shape[1]-1) 
        y= min(y, image.shape[0]-1)
        #Check if the patch is out of bounds
        if (x - radius < 0) or (x + radius >= image.shape[1]) or (y - radius < 0) or (y + radius >= image.shape[0]):
            #Add zero padding
            image = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)

            #update x,y
            x += radius
            y += radius
        #landmarks[i,:] = x,y
        keypoints.append(cv2.KeyPoint(float(x), float(y), radius))

    sift = cv2.SIFT_create()
    result_keypoints, descriptors = sift.compute(image, keypoints)

    if len(descriptors) != len(landmarks):
        print("ERROR: descriptors and landmarks don't match")
        return None

    for i in range(len(landmarks)):
        if descriptors is not None and descriptors[i] is not None:
            features[i]=descriptors[i].flatten()
  
    return features.flatten()


def getORBFromLandmarks(landmarks, image, radius=10, preprocess=True):

    features = np.zeros((len(landmarks), 32))

    if preprocess:
        image = processImage(image) # REMOVE AND DO ONCE

    keypoints = []

    for i in range(len(landmarks)):
        # Extract the patch around the landmark
        x, y = landmarks[i]
        x= max(0,x) 
        y= max(0,y)
        x= min(x, image.shape[1]-1) 
        y= min(y, image.shape[0]-1)
        #Check if the patch is out of bounds
        if (x - radius < 0) or (x + radius >= image.shape[1]) or (y - radius < 0) or (y + radius >= image.shape[0]):
            #Add zero padding
            image = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)

            #update x,y
            x += radius
            y += radius
        #landmarks[i,:] = x,y
        keypoints.append(cv2.KeyPoint(float(x), float(y), radius))

    orb = cv2.ORB_create()

    result_keypoints, descriptors = orb.compute(image, keypoints)
    result_keypoints = [keypoint.pt for keypoint in result_keypoints]
    padded_descriptors = np.zeros((len(landmarks), 32))
    #check which keypoints doesn't have descriptors and add zero descriptors
    if len(descriptors) != len(landmarks):
        for i in range(len(landmarks)):
            if keypoints[i].pt not in result_keypoints:
                padded_descriptors[i] = np.zeros(32)
            else:
                indexOfDescriptor = result_keypoints.index(keypoints[i].pt)
                padded_descriptors[i] = descriptors[indexOfDescriptor]

    if len(padded_descriptors) != len(landmarks):
        print("ERROR: descriptors and landmarks don't match")
        return None

    for i in range(len(landmarks)):
        if padded_descriptors is not None and padded_descriptors[i] is not None:
            features[i]=padded_descriptors[i].flatten()
  
    return features.flatten()