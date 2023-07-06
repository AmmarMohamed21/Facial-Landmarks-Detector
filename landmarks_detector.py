from enum import Enum
import os
import pandas as pd
import numpy as np
import ast
import joblib 
import cv2
from image_features_extractor import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


#Indices of eyes used in calculating normalized error
rightEyeStartIndex = 36 
rightEyeEndIndex = 39
leftEyeStartIndex = 42
leftEyeEndIndex = 45


class FeatureType(Enum):
    '''
    describing the type of features used in training the regressors
    '''
    HOG = 1
    SIFT = 2
    ORB = 3

class LandmarksDetector:
    def __init__(self, isPredictor=False, modelspath=None):
        '''
        isPredictor: if true, load models from modelspath for using the model for prediction
        modelspath: path to folder containing regressor models, pca models, standard scaler models and mean shape
        '''

        #Loading models from modelspath and initial mean shape
        if isPredictor:
            assert modelspath is not None

            self.loadedregressors = []
            self.loadedStandardModels = []
            self.loadedPCAModels = []
            for index in range(0,9):
                filename = modelspath+'/regressor'+str(index)+'.pkl'
                self.loadedregressors.append(joblib.load(filename))

            for index in range(0,3):
                filename = modelspath+'/scaler'+str(index)+'.pkl'
                self.loadedStandardModels.append(joblib.load(filename))

            for index in range(0,3):
                filename = modelspath+'/pca'+str(index)+'.pkl'
                self.loadedPCAModels.append(joblib.load(filename))
            
            self.mean_shape = np.load(modelspath+'/mean_shape.npz')['shape'].reshape(68,2).round().astype(int)

            self.imagesmodelshape = (200,200)
        
    
    def predict(self, image, facebounds):
        '''
        Performs landmarks detection on image 
        image: input np array image, can be either grayscale, rgb or bgr
        facebounds: bounds of face in image (x1,y1,x2,y2)
        '''
        faceimage = image[facebounds[1]:facebounds[3], facebounds[0]:facebounds[2]]
                
        #set initial shape to mean shape
        initial_prediction = self.mean_shape

        #originalsize of face
        originalFaceSize = faceimage.shape

        #resize image to (200,200)
        img = cv2.resize(faceimage, self.imagesmodelshape)

        #performs needed preprocessing, transform to gray scale if needed and histogram equalization
        img = processImage(img)

        #initial prediction
        shape = initial_prediction

        #Applying all 9 regressors to get final shape
        shapesresults = []
        shapesresults.append(shape)
        for i in range (0, 3):
            hog_features = getHogFromLandmarks(shape, img, preprocess=False)
            offset = self.loadedregressors[i].predict([hog_features])
            shape = shape + offset.reshape((68,2))
            shape = shape.round().astype(int)
            shapesresults.append(shape)

        for i in range (3, 6):
            hog_features = getHogFromLandmarks(shape, img, preprocess=False)
            offset = self.loadedregressors[i].predict([hog_features])
            shape = shape + offset.reshape((68,2))
            shape = shape.round().astype(int)
            shapesresults.append(shape)

        for i in range (6, 9):
            sift_feutures = getSiftFromLandmarks(shape, img, preprocess=False)
            sift_features = self.loadedStandardModels[i-6].transform([sift_feutures])
            pca_features = self.loadedPCAModels[i-6].transform(sift_features)
            offset = self.loadedregressors[i].predict(pca_features)
            shape = shape + offset.reshape((68,2))
            shape = shape.round().astype(int)
            shapesresults.append(shape)


        landmarks = shape

        #resize landmarks to original size on the original image
        landmarks = landmarks * (np.array(originalFaceSize)/np.array(self.imagesmodelshape))
        landmarks = landmarks.round().astype(int)

        landmarks = landmarks + np.array([facebounds[0],facebounds[1]])

        return landmarks

    def train(self, df_path, images_train_path, resultsOutputPath='models', L=3, K=3, sampleSize=5, alphas=[10000,5000,40000], B=1, T=10, featuresUsed=[FeatureType.HOG, FeatureType.HOG, FeatureType.SIFT]):
        '''
        df_path: path to csv file containing training set
        images_train_path: path to folder containing training face images
        resultsOutputPath: path to folder where models will be saved, default is 'models', it creates a folder for the models in the output directoy
        L: number of iterations or stages for CFSS Algorithm
        K: number of regressors for each stage
        sampleSize: number of samples to be sampled at the start of each iteration
        alphas: regularization parameters for each stage, should be of size L
        B: sensitivty parameter for similarity measure
        T: number of iterations for weights update
        featuresUsed: list of features described as enum to be used for each regressor, should be of size L
        '''
        self.__initializeReporting(resultsOutputPath)
        self.__setHyperParameters(L, K, sampleSize, alphas, B, T, featuresUsed)
        self.__prepareDataset(df_path, images_train_path)
        regressors, standardScalarModels, pcaModels = self.__runTraining()

        for index,reg in enumerate(regressors):
            filename = self.resultsOutputPath+'/regressor'+str(index)+'.pkl'
            joblib.dump(reg, filename)

        for index,pca in enumerate(pcaModels):
            filename = self.resultsOutputPath+'/pca'+str(index)+'.pkl'
            joblib.dump(pca, filename)

        for index,scaler in enumerate(standardScalarModels):
            filename = self.resultsOutputPath+'/scaler'+str(index)+'.pkl'
            joblib.dump(scaler, filename)

        self.report.close()
        return regressors, standardScalarModels, pcaModels, self.x_bar_initial

    
    def __runTraining(self):
        '''
        Main traning function that runs the CFSS algorithm
        '''
        regressors = []
        pcaModels = []
        standardScalarModels = []

        for l in range(self.L):
            # Your code here
            self.report.write("Iteration "+str(l+1)+"\n")
            print("Iteration "+str(l+1)+"\n")

            withreplacement = False #True if l == 2 else False

            #For each image in candidate_shapes, we sample possible shapes to get a result shape of (candidate_shapes.shape[0],sampleSize,candidate_shapes.shape[1])
            samples = np.array([self.__getSamplesForEachImage(self.candidate_shapes, sampleSize=self.sampleSize, probabilities=self.probabilities[i], replace=withreplacement) for i in range(0,self.candidate_shapes.shape[0])])

            #samplesList.append(samples)
            # if l==2:
            #     randomError = np.random.randint(-3,3, size=samples.shape).astype(int)
            #     samples +=randomError.round()

            #Calculate Error at beginning of iteration after sampling
            error = self.__calc_total_error(samples,self. resized_ground_truth)
            self.report.write("Iteration "+str(l+1)+" Error after sampling: "+ str(error)+'\n')  
            print("Iteration "+str(l+1)+" Error after sampling: "+ str(error)+'\n')       


            for k in range(self.K):

                print("K: ", k+1)
                self.report.write("K: "+str(k+1)+"\n")
                
                #calculating labels for each sample where the label is the difference between the sample and the ground truth of the corresponing image
                labels =self. candidate_shapes[:,np.newaxis,:] - samples
                assert (labels[self.sampleSize,3,:] == (self.candidate_shapes[self.sampleSize,:] - samples[self.sampleSize,3,:])).all()
                labels = labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])

                print("Feature Extraction...")
                #startfeatures = time.time()
                #Now we calculate the feature vector for each sample shape using hog function getHogFromLandmarks
                if self.featuresUsed[l] == FeatureType.HOG:
                    print("HOG")
                    features = np.array([[getHogFromLandmarks(samples[i,j].reshape(68,2), cv2.imread(self.images_train_path+self.images_train[i])) for j in range(0,self.sampleSize)] for i in range(0,self.candidate_shapes.shape[0])])
                    features = features.reshape(features.shape[0]*features.shape[1],features.shape[2])
                elif self.featuresUsed[l] == FeatureType.ORB:
                    features = np.array([[getORBFromLandmarks(samples[i,j].reshape(68,2), cv2.imread(self.images_train_path+self.images_train[i])) for j in range(0,self.sampleSize)] for i in range(0,self.candidate_shapes.shape[0])])
                    features = features.reshape(features.shape[0]*features.shape[1],features.shape[2])
                elif self.featuresUsed[l] == FeatureType.SIFT:
                    print("Using SIFT")
                    features = np.array([[getSiftFromLandmarks(samples[i,j].reshape(68,2), cv2.imread(self.images_train_path+self.images_train[i])) for j in range(0,self.sampleSize)] for i in range(0,self.candidate_shapes.shape[0])])
                    features = features.reshape(features.shape[0]*features.shape[1],features.shape[2])
                    #Apply pca on features
                    print("standardization")
                    scaler = StandardScaler()
                    features = scaler.fit_transform(features)
                    standardScalarModels.append(scaler)

                    print("PCA")
                    pca = PCA(n_components=2176)
                    features = pca.fit_transform(features)
                    pcaModels.append(pca)

                    self.report.write("PCA components: "+str(features.shape[1])+"\n")
                    print("PCA components: ", features.shape[1])

                #endfeatures = time.time()
                #print("Feature Extraction time: ", endfeatures-startfeatures)

                assert features.shape[0] == labels.shape[0]

                print("Training regressor...")

                #startraining = time.time()
                #Train regressor with l2 regularization
                usedalpha = self.alphas[l]
                reg = Ridge(usedalpha).fit(features, labels)

                self.report.write("Alpha: "+str(usedalpha)+"\n")
                print("Alpha: ", usedalpha)
                #endtraining = time.time()
                #print("Training time: ", endtraining-startraining)

                regressors.append(reg)

                #update samples with the offset calculated from predicted values
                predicted_labels = reg.predict(features)
                predicted_labels = predicted_labels.reshape(self.candidate_shapes.shape[0],self.sampleSize,self.candidate_shapes.shape[1])
                samples = samples + predicted_labels
                #regressedShapes.append(samples)
                #round samples to integers
                samples = np.round(samples).astype(int)

                #startError = time.time()
                trainingError = self.__calc_total_error(samples, self.resized_ground_truth)
                #endError = time.time()
                #print("Error calculation time: ", endError-startError)

                #normalizedErrors.append(trainingError)

                #Print total normalized error
                print("Error after iteration l=",l+1,", regressor k=",k+1,": ",trainingError)
                self.report.write("Error after iteration l="+str(l+1)+", regressor k="+str(k+1)+": "+str(trainingError)+'\n')

            
            if l < self.L-1:
                print("Updating distributions...")
                #startDistribution = time.time()
                #Create a weight vector of length sampleSize for each image, initially set to e/SampleSize
                weights = np.ones((self.candidate_shapes.shape[0],self.sampleSize, 1)) * (1/self.sampleSize)

                #Updating weights
                updated_weights = self.__UpdateWeights(weights, samples)

                #Update x_bar using updated weights and regressed samples
                x_bar = np.zeros((self.candidate_shapes.shape[0],self.candidate_shapes.shape[1]))
                for index,weight in enumerate(updated_weights):
                    x_bar[index] = np.matmul(weight.T, samples[index])

                self.probabilities = np.array([self.__calculateDistribution(x_bar[i]) for i in range(0,x_bar.shape[0])])

                #endDistribution = time.time()
                #print("Distribution update time: ", endDistribution-startDistribution)
            
            #enditeraton = time.time()

            #print("Time for iteration: ",l+1, enditeraton-startiteration)
        return regressors, standardScalarModels, pcaModels
        
    def __prepareDataset(self, df_path, images_train_path):
        '''
        Initializes dataset for training, setting candidate shapes and mean shape
        '''
        df = pd.read_csv(df_path)
        landmarkslist = df['landmarks'].values.tolist()
        self.landmarks_dataset_train = np.array([ast.literal_eval(x) for x in landmarkslist])
        self.images_train_path = images_train_path

        #reshape landmarks to one vector
        self.landmarks_dataset_train = self.landmarks_dataset_train.reshape(self.landmarks_dataset_train.shape[0],self.landmarks_dataset_train.shape[1]*2)

        self.candidate_shapes = self.landmarks_dataset_train

        self.resized_ground_truth = np.repeat(self.candidate_shapes,self.sampleSize,axis=0).reshape(self.candidate_shapes.shape[0],self.sampleSize,self.candidate_shapes.shape[1])


        self.x_bar_initial = self.candidate_shapes.mean(axis=0)
        np.savez_compressed(self.resultsOutputPath+'/mean_shape.npz', shape=self.x_bar_initial)
        #Set x_bar to be the same for all images as x_bar_initial, x_bar shape is (imagesSize, x_bar_initial.shape[0])
        self.x_bar = np.tile(self.x_bar_initial, (self.candidate_shapes.shape[0],1))

        #initial Probability distrubtions, set as uniform, shape is (imagesSize, imagesSize)
        self.probabilities = np.ones((self.candidate_shapes.shape[0],self.candidate_shapes.shape[0]))/self.candidate_shapes.shape[0]

        self.images_train = df['images'].values.tolist()

        self.report.write('trainLength = '+str(len(self.images_train))+'\n')
        self.report.write('imagesPath = '+self.images_train_path+'\n\n')
        self.report.write("Models path: "+self.resultsOutputPath+'\n')
        self.report.write("X_bar path: "+'mean_shape.npz'+'\n\n')



    def __setHyperParameters(self, L, K, sampleSize, alphas, B, T, featuresUsed):
        '''
        Sets hyperparameters for the algorithm and prints them to the report
        '''
        self.L = L
        self.K = K
        self.sampleSize = sampleSize
        self.alphas = alphas
        self.B = B
        self.T = T
        self.featuresUsed = featuresUsed
        assert len(self.featuresUsed) == self.L
        assert len(self.alphas) == self.L

        self.report.write('Parameters:\n')

        self.report.write('L = '+str(L)+'\n')
        self.report.write('K = '+str(K)+'\n')
        self.report.write('sampleSize = '+str(sampleSize)+'\n')
        self.report.write('alphas = '+str(alphas)+'\n')
        self.report.write('Features Used = '+str(featuresUsed)+'\n')
        self.report.write('B = '+str(B)+'\n')
        self.report.write('T = '+str(T)+'\n')
    
    def __initializeReporting(self, resultsOutputPath):
        '''
        Initializes reporting, creates folder for results and report file
        report is used for each training trial to analyze results
        '''
        if not os.path.exists(resultsOutputPath):
            os.makedirs(resultsOutputPath)
        numModels = len(os.listdir(resultsOutputPath))
        self.resultsOutputPath = resultsOutputPath+'/model'+str(numModels)
        os.makedirs(self.resultsOutputPath)
        reportName = self.resultsOutputPath+'/report.txt'
        self.report = open(reportName, "w")
        self.report.write("Models path: "+self.resultsOutputPath+'\n')

    def __calculateDistribution(self, x_bar):
        '''
        Calculates Probability distribution of candidate shape for each training image using x_bar
        Distribution is calculated using gaussian similarity function
        x_bar: mean shape of samples after each stage of training size (trainingSize, 136)
        '''
        differences = np.tile(x_bar, (len(self.candidate_shapes), 1)) - self.candidate_shapes
        cov =  np.diag(np.var(differences, axis=0))
        inv_cov = np.linalg.inv(cov)
        temp = np.matmul(differences, inv_cov)

        exponent = -0.5 * np.sum(temp * differences, axis=1)
        distribution = np.exp(exponent)
        #normalize distribution
        distribution = distribution / np.sum(distribution)
        return distribution
    
    def __UpdateWeights(self,weights, samples):
        '''
        Updating weights of samples using affinity matrix A and Replicator Dynamics
        '''
        AffinityMatrices = self.__CalcAffinityMatrix(samples)
        for t in range(self.T):
            for index,A in enumerate(AffinityMatrices):
                weights[index] = (np.multiply(weights[index], np.dot(A, weights[index]))) / (np.matmul(weights[index].T, np.dot(A, weights[index])))
        return weights

    
    def __CalcAffinityMatrix(self, samples): # B is the inverse of the variance, higher B means more sensitive to differences
        '''
        Function to calculate affinity matrix A for each image such that A is of size (sampleSize,sampleSize)
        '''
        AffinityMatrices = np.zeros((samples.shape[0],self.sampleSize,self.sampleSize))
        for index,sampleVector in enumerate(samples):
            A = np.zeros((self.sampleSize,self.sampleSize))
            for i in range(0,self.sampleSize):
                for j in range(0,self.sampleSize):
                    A[i,j] = 0 if i==j else np.exp(-self.B * np.linalg.norm(sampleVector[i]-sampleVector[j]))
            AffinityMatrices[index] = A
        return AffinityMatrices

    def __getSamplesForEachImage(self, candidate_shapes, probabilities, sampleSize=5, replace=False):
        '''
        Sampling from candidate shapes for each image given the probability distribution and sample Size
        '''
        np.random.seed(21)
        rng = np.random.default_rng()
        sampled_shapes =rng.choice(candidate_shapes, sampleSize, replace=replace, p=probabilities) 
        return sampled_shapes

    #Calculate normalized error between predicted shape and ground truth
    def __calc_normalized_error_single_image(self, predicted_shape, ground_truth):
        '''
        Calculating inter-ocular normalized error For Single Image
        predicted_shape: shape of size (136)
        ground_truth: shape of size (136)
        '''
        predicted_shape = predicted_shape.reshape(-1,2)
        ground_truth = ground_truth.reshape(-1,2)
        leftPupilCoordinates = ((ground_truth[leftEyeStartIndex]+ground_truth[leftEyeEndIndex])/2).round().astype(int)
        rightPupilCoordinates = ((ground_truth[rightEyeStartIndex]+ground_truth[rightEyeEndIndex])/2).round().astype(int)
        interpupilary_distance = np.linalg.norm(leftPupilCoordinates-rightPupilCoordinates)

        #normalized error is the euclidean distance between the predicted shape and the ground truth divided by the interocular distance
        error = np.average(np.linalg.norm(predicted_shape-ground_truth, axis=1)/interpupilary_distance)
        return error

    #For list of images
    def __calc_total_error(self, predicted_shapes, ground_truths):
        '''
        calculating total error for a list of images given
        '''
        assert predicted_shapes.shape == ground_truths.shape

        totalError = 0
        for i in range(0,len(predicted_shapes)):
            totalError += self.__calc_normalized_error_single_image(predicted_shapes[i], ground_truths[i])
        return totalError/len(predicted_shapes)
    
    def calculateTestAccuracy(self, testfile_path, imagestest_path, x_bar_initial, regressors, featuresUsed, pcamodels=None, standardModels=None):
        '''
        Calculate test accuracy for a given dataset
        testfile_path: path to csv file containing image names and landmarks
        imagestest_path: path to folder containing images
        x_bar_initial: mean shape of training data
        regressors: list of regressors trained at each stage of training
        featuresUsed: list of features used at each stage of training
        pcamodels (optional): list of pca models trained for sift
        standardModels (optional): list of standard models trained for sift
        '''
        #check if report is closed
        if self.report.closed:
            reportName = self.resultsOutputPath+'/report.txt'
            self.report = open(reportName, "a")
        df_test = pd.read_csv(testfile_path)
        images_test = df_test['images'].values
        landmarkslist_test = df_test['landmarks'].values.tolist()
        landmarks_test = np.array([ast.literal_eval(x) for x in landmarkslist_test])
        landmarks_test = landmarks_test.reshape(landmarks_test.shape[0],landmarks_test.shape[1]*2)

        x_bar_initial = self.x_bar_initial.reshape(68,2).round().astype(int)
        x_bar = np.tile(x_bar_initial, (landmarks_test.shape[0],1))

        predicted_shapes = x_bar
        predicted_shapes = predicted_shapes.reshape((landmarks_test.shape[0],68,2))

        
        L_stages = int(len(regressors)/3)
        assert len(featuresUsed) == L_stages

        siftStartIndex=None
        for l in range(L_stages):
            if FeatureType.SIFT == featuresUsed[l]:
                siftStartIndex = l*3
                break

        #for i in range(len(landmarks_test)):
            #predicted_shape = x_bar
        for index, regressor in enumerate(regressors):
            #extract hog features from predicted_shape
            if pcamodels is not None and featuresUsed[index//3] == FeatureType.SIFT:
                features = np.array([getSiftFromLandmarks(predicted_shapes[i], cv2.imread(imagestest_path+images_test[i])) for i in range(0, predicted_shapes.shape[0])]) 
                pcaIndex = 0 if siftStartIndex == 0 else index%siftStartIndex
                features = standardModels[pcaIndex].transform(features)
                features = pcamodels[pcaIndex].transform(features)
            elif featuresUsed[index//3] == FeatureType.ORB:
                features = np.array([getORBFromLandmarks(predicted_shapes[i], cv2.imread(imagestest_path+images_test[i])) for i in range(0, predicted_shapes.shape[0])]) 
            else:
                features = np.array([getHogFromLandmarks(predicted_shapes[i], cv2.imread(imagestest_path+images_test[i])) for i in range(0, predicted_shapes.shape[0])]) 


            #hog_features = getSiftFromLandmarks(predicted_shape, img)
            offset = regressor.predict(features)
            predicted_shapes = predicted_shapes + offset.reshape((landmarks_test.shape[0],68,2))
            predicted_shapes = predicted_shapes.round().astype(int)
            #predicted_shapes.append(predicted_shape)
        acc =  self.__calc_total_error(predicted_shapes.reshape(predicted_shapes.shape[0], predicted_shapes.shape[1]*predicted_shapes.shape[2]), landmarks_test)
        self.report.write("Test Accuracy on "+imagestest_path.split('/')[0]+" images: "+str(acc)+"\n")
        print("Test Accuracy on "+imagestest_path.split('/')[0]+" images: "+str(acc)+"\n")
        self.report.close()