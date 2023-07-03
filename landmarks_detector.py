from enum import Enum
import os
import pandas as pd
import numpy as np
import ast

#right eye start is 38
rightEyeStartIndex = 38
#right eye end is 42
rightEyeEndIndex = 42
#left eye start is 48
leftEyeStartIndex = 48
#left eye end is 51
leftEyeEndIndex = 51

class FeatureType(Enum):
    HOG = 1
    SIFT = 2
    BRIEF = 3

class LandmarksDetector:
    def __init__(self):
        pass #TODO

    def train(self, df_path, images_path, resultsOutputPath='models', L=2, K=3, sampleSize=15, alpha=1000, alpha_sift=100000, B=1, T=10, featuresUsed=[FeatureType.HOG, FeatureType.SIFT]):

        self.__initializeReporting(resultsOutputPath)
        self.__setHyperParameters(L, K, sampleSize, alpha, alpha_sift, B, T, featuresUsed)

        
    def __prepareDataset(self, df_path, images_path):
        df = pd.read_csv(df_path)
        landmarkslist = df['landmarks'].values.tolist()
        landmarks_dataset = np.array([ast.literal_eval(x) for x in landmarkslist])

        #reshape landmarks to one vector
        landmarks_dataset = landmarks_dataset.reshape(landmarks_dataset.shape[0],landmarks_dataset.shape[1]*2)

        #Split to train and test
        trainLength = 2000
        landmarks_train = landmarks_dataset[:trainLength]
        images_train = df['images'].values.tolist()[:trainLength]
        landmarks_test = landmarks_dataset[trainLength:]
        images_test = df['images'].values.tolist()[trainLength:]

        candidate_shapes = landmarks_train

        resized_ground_truth = np.repeat(candidate_shapes,sampleSize,axis=0).reshape(candidate_shapes.shape[0],sampleSize,candidate_shapes.shape[1])


        x_bar_initial = candidate_shapes.mean(axis=0)
        #Set x_bar to be the same for all images as x_bar_initial, x_bar shape is (imagesSize, x_bar_initial.shape[0])
        x_bar = np.tile(x_bar_initial, (candidate_shapes.shape[0],1))

        #initial Probability distrubtions, set as uniform, shape is (imagesSize, imagesSize)
        probabilities = np.ones((candidate_shapes.shape[0],candidate_shapes.shape[0]))/candidate_shapes.shape[0]

    def __setHyperParameters(self, L, K, sampleSize, alpha, alpha_sift, B, T, featuresUsed):
        self.L = L
        self.K = K
        self.sampleSize = sampleSize
        self.alpha = alpha
        self.alpha_sift = alpha_sift
        self.B = B
        self.T = T
        self.featuresUsed = featuresUsed
        assert len(self.featuresUsed) == self.L

        self.report.write('Parameters:\n')

        self.report.write('L = '+str(L)+'\n')
        self.report.write('K = '+str(K)+'\n')
        self.report.write('sampleSize = '+str(sampleSize)+'\n')
        self.report.write('alpha = '+str(alpha)+'\n')
        self.report.write('alpha_sift = '+str(alpha_sift)+'\n')
        self.report.write('Features Used = '+str(featuresUsed)+'\n')
        self.report.write('B = '+str(B)+'\n')
        self.report.write('T = '+str(T)+'\n')
    
    def __initializeReporting(self, resultsOutputPath):
        if not os.path.exists(resultsOutputPath):
            os.makedirs(resultsOutputPath)
        numModels = len(os.listdir(resultsOutputPath))
        resultsOutputPath = resultsOutputPath+'/model'+str(numModels)
        os.makedirs(resultsOutputPath)
        reportName = resultsOutputPath+'/report.txt'
        self.report = open(reportName, "w")
        self.report.write("Models path: "+resultsOutputPath+'\n')
