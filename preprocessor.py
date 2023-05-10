import os 
import pandas as pd
import ast
import numpy as np
import cv2

def CreateCSVFromAnnotations(annotationsPath):
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
    df.to_csv('/content/drive/My Drive/helen.csv', index=False)


def array_to_columns(row):
    new_row = {}
    landmarks = ast.literal_eval(row['landmarks'])
    reducedLandmarks = []
    for i in range(0, len(landmarks), 3):
        reducedLandmarks.append([landmarks[i][0],landmarks[i][1]])
    new_row['landmarks'] = reducedLandmarks
    return pd.Series(new_row)


def ReduceHelenDataset(datasetPath, targetDatasetPth):
    df = pd.read_csv(datasetPath)
    new_df = df.apply(array_to_columns, axis=1)
    new_df.columns = ['landmarks']
    df.drop('landmarks', axis=1, inplace=True)
    new_df = pd.concat([df, new_df], axis=1)
    new_df.to_csv(targetDatasetPth, index=False)
    

def procrustes_analysis(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
   
    return d, Z, tform


def CropAndResizeHelen():
    '''
    This function crops the images in the helen dataset around the face and resizes them to 400x400.
    ''' 
    df = pd.read_csv('datasets/helen.csv')
    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        # Get image name and landmarks

        image_name = row['images']
        landmarks = ast.literal_eval(row['landmarks'])

        image_dir = 'datasets/helen/'
        
        # Load image
        img = cv2.imread(image_dir + image_name)
        
        # Convert landmarks to numpy array
        landmarks = np.array(landmarks)
        
        # Extract facial landmarks
        x1, y1 = np.min(landmarks, axis=0)
        x2, y2 = np.max(landmarks, axis=0)
        
        # Compute the center of the face
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Compute the width and height of the face
        w, h = x2 - x1, y2 - y1
        
        # Add some padding to the face region
        padding = 0.3
        x1 -= int(w * padding)
        y1 -= int(h * padding)
        x2 += int(w * padding)
        y2 += int(h * padding)

        # Make sure the padding is not out of bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1] - 1), min(y2, img.shape[0] - 1)

        # Make the bounding box a square
        w, h = x2 - x1, y2 - y1
        if w > h:
            x1 += (w - h) // 2
            x2 -= (w - h) // 2
        else:
            y1 += (h - w) // 2
            y2 -= (h - w) // 2

        
        # Crop the image around the face
        face_img = img[y1:y2, x1:x2]

        # Resize the image to 400x400
        targetShape = (400, 400)
        face_img = cv2.resize(face_img, targetShape)
        
        # Update landmarks to match the cropped image and resized image
        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1
        landmarks = landmarks * targetShape[0] // (x2 - x1)    
        
        # Save the cropped image
        cv2.imwrite('datasets/croppedHelen/' + image_name, face_img)
        
        # Update the landmarks in the dataframe
        df.at[index, 'landmarks'] = landmarks.tolist()

    df.to_csv('datasets/croppedHelen.csv', index=False)