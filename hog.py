import numpy as np
import cv2

class Hog:
    def __sobel(self,image):
        paddedImg = np.pad(image, ((1, 1), (1, 1)), 'edge')
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        gx = cv2.filter2D(paddedImg, -1, sobel_x)[1:-1, 1:-1]
        gy = cv2.filter2D(paddedImg, -1, sobel_y)[1:-1, 1:-1]
        
        return gx,gy
    def __compute_gradients(self, image):
        gx,gy = self.__sobel(image)
        gradients = np.sqrt(gx**2 + gy**2)
        angles = np.arctan2(gy, gx) * (180/np.pi)
        return gradients, angles
    
    def __compute_local_orietnation_hists(self, grads, angles, pixels_per_cell=(10,10), orientations=8):
        hists = np.zeros((grads.shape[0]//pixels_per_cell[0], grads.shape[1]//pixels_per_cell[1], orientations))
        for i in range(0, angles.shape[0]-pixels_per_cell[0]+1, pixels_per_cell[0]):
            for j in range(0, grads.shape[1]-pixels_per_cell[1]+1, pixels_per_cell[1]):
                cell = angles[i:i+pixels_per_cell[0], j:j+pixels_per_cell[1]]
                hist, _ = np.histogram(cell, bins=orientations, range=(-180, 180), weights=grads[i:i+pixels_per_cell[0], j:j+pixels_per_cell[1]])
                hists[i//pixels_per_cell[0], j//pixels_per_cell[1]] = hist
        return hists
    
    def __compute_block_norms(self, hists, cells_per_block=(2,2)):
        block_norms = np.zeros((hists.shape[0]-cells_per_block[0]+1, hists.shape[1]-cells_per_block[1]+1, cells_per_block[0]*cells_per_block[1]*hists.shape[2]))
        for i in range(0, hists.shape[0]-cells_per_block[0]+1):
            for j in range(0, hists.shape[1]-cells_per_block[1]+1):
                block = hists[i:i+cells_per_block[0], j:j+cells_per_block[1]]
                block_norms[i, j] = block.ravel() / np.sqrt(np.sum(block**2) + 1e-5)
        feature_vector = block_norms.flatten()
        return feature_vector
    
    def CalculateHog(self, image, pixels_per_cell=(10,10), cells_per_block=(2,2), orientations=8):
        gradients, angles = self.__compute_gradients(image/255.0)
        hists = self.__compute_local_orietnation_hists(gradients, angles, pixels_per_cell=pixels_per_cell, orientations=orientations)
        feature_vector = self.__compute_block_norms(hists, cells_per_block=cells_per_block)
        return feature_vector