import os 
import pandas as pd
import ast
import numpy as np
import cv2

def CreateCSVFromAnnotations68(annotationsPath, outputPath):
    '''
    Given a folder contains .pts files annotating the landmarks of the images using ibug format
    outputs a csv file with the following format: images, landmarks
    '''
    images = []
    landmarks = []
    allfiles = os.listdir(annotationsPath)
    for filename in allfiles:
      #read only files with extension .pts
        if filename.endswith(".pts"):
            f = os.path.join(annotationsPath, filename)
            if os.path.isfile(f):
                with open(f) as file:
                    #check if image is jpg or png
                    if filename.split(".")[0]+".jpg" in allfiles:
                        images.append(filename.split(".")[0]+".jpg")
                    else:
                        images.append(filename.split(".")[0]+".png")
                    #Skip first 3 lines
                    for i in range(3):
                        file.readline()
                    
                    imagelabels = []
                    i = 0
                    for line in file: 
                        if i == 68:
                            break
                        cooardinates = line.split(" ")
                        cooardinates = [int(round(float(cooardinates[0]))), int(round(float(cooardinates[1])))]
                        imagelabels.append(cooardinates)
                        i+=1
                    landmarks.append(imagelabels)

    df = pd.DataFrame(data = {'images':images, 'landmarks': landmarks})
    df.to_csv(outputPath, index=False)

def CreateCSVFromAnnotations(annotationsPath, outputPath):
    '''
    Given a folder contains images and .txt files annotating the landmarks of the images using helen format
    outputs a csv file with the following format: images, landmarks
    '''
    images = []
    landmarks = []
    for filename in os.listdir(annotationsPath):
      f = os.path.join(annotationsPath, filename)
      if os.path.isfile(f):
        with open(f) as file:
            images.append(file.readline().split("\n")[0]+".jpg")
            imagelabels = []
            for line in file:  #for line_num,line in enumerate(file):
                cooardinates = line.split(" , ")
                cooardinates = [int(round(float(cooardinates[0]))), int(round(float(cooardinates[1])))]
                imagelabels.append(cooardinates)
            landmarks.append(imagelabels)
    df = pd.DataFrame(data = {'images':images, 'landmarks': landmarks})
    df.to_csv(outputPath, index=False)


def array_to_columns(row):
    '''
    Used to reduce helen 194 landmarks to 65 landmarks by keeping every 3rd landmark
    '''
    new_row = {}
    landmarks = ast.literal_eval(row['landmarks'])
    reducedLandmarks = []
    for i in range(0, len(landmarks), 3):
        reducedLandmarks.append([landmarks[i][0],landmarks[i][1]])
    new_row['landmarks'] = reducedLandmarks
    return pd.Series(new_row)


def ReduceHelenDataset(datasetPath, targetDatasetPth):
    '''
    Used to reduce helen 194 landmarks to 65 landmarks by keeping every 3rd landmark
    '''
    df = pd.read_csv(datasetPath)
    new_df = df.apply(array_to_columns, axis=1)
    new_df.columns = ['landmarks']
    df.drop('landmarks', axis=1, inplace=True)
    new_df = pd.concat([df, new_df], axis=1)
    new_df.to_csv(targetDatasetPth, index=False)


def CropAndResizeDataset(filename='datasets/helen.csv', srcImageDir='datasets/helen/', targetImagesDir = 'datasets/croppedHelen2/', targetCSVName= 'datasets/croppedHelen2.csv',targetShape=(200,200), cropPaddingTop=0.3, cropPaddingBottom=0.1, cropPaddingLeft=0.1, cropPaddingRight=0.1):
    '''
    This function crops the images in the helen dataset around the face and resizes them to input target shape.
    It also creates a new csv file with the following format: images, landmarks

    filename: path to the csv file containing the images and landmarks
    srcImageDir: path to the folder containing the images
    targetImagesDir: path to the folder where the cropped images will be saved
    targetCSVName: name of the csv file to be created
    targetShape: shape of the cropped images
    cropPaddingTop: padding to be added to the top of the face region
    cropPaddingBottom: padding to be added to the bottom of the face region
    cropPaddingLeft: padding to be added to the left of the face region
    cropPaddingRight: padding to be added to the right of the face region
    ''' 
    df = pd.read_csv(filename)
    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        # Get image name and landmarks

        image_name = row['images']
        landmarks = ast.literal_eval(row['landmarks'])

        image_dir = srcImageDir
        
        # Load image
        img = cv2.imread(image_dir + image_name)
        
        # Convert landmarks to numpy array
        landmarks = np.array(landmarks)
        
        # Extract facial landmarks
        x1, y1 = np.min(landmarks, axis=0)
        x2, y2 = np.max(landmarks, axis=0)
        
        # Compute the width and height of the face
        w, h = x2 - x1, y2 - y1
        
        # Add some padding to the face region
        x1 -= int(w * cropPaddingLeft)
        y1 -= int(h * cropPaddingTop)
        x2 += int(w * cropPaddingRight)
        y2 += int(h * cropPaddingBottom)

        # Make sure the padding is not out of bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1] - 1), min(y2, img.shape[0] - 1)

        # Make the bounding box a square #TODO
        w, h = x2 - x1, y2 - y1
        if w > h:
            x1 += (w - h) // 2
            x2 -= (w - h) // 2
        else:
            y1 += (h - w) // 2
            y2 -= (h - w) // 2

        
        # Crop the image around the face
        face_img = img[y1:y2, x1:x2]

        # Resize the image to targetShape
        face_img = cv2.resize(face_img, targetShape)
        
        # Update landmarks to match the cropped image and resized image
        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1
        landmarks = landmarks * targetShape[0] // (x2 - x1)    
        
        # Save the cropped image
        #check if directory exists
        if not os.path.exists(targetImagesDir):
            os.makedirs(targetImagesDir)
        cv2.imwrite(targetImagesDir + image_name, face_img)
        
        # Update the landmarks in the dataframe
        df.at[index, 'landmarks'] = landmarks.tolist()

    df.to_csv(targetCSVName, index=False)