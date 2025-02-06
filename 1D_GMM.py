# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:19:47 2025

@author: BIDISHA
"""

import numpy as np
import cv2
import os
import time
# Initial values
K = 4 # No. of Gaussians
init_pi = [0.7,0.11,0.1,0.09] # weights of Gaussians
init_mu = 0 # mean of Gaussians
init_sigma = 255 # variance of Gaussians
init_alpha = 0.05 # learning rate
threshold = 0.9 # For selecting 'B' background Gaussians

# Convert the image of the first frame from RGB format to HSV format and extract the V channel that corresponds to the grayscale image
img = cv2.cvtColor(cv2.imread(r"C:\Users\BIDISHA\Desktop\ELL784_Assignment1\Frames\0.bmp"), cv2.COLOR_BGR2HSV)[:,:,0]
img_shape = img.shape # extracts the height and width of the grayscale image
print(img_shape)

# Modelling each pixel as a mixture of Gaussians
pi = np.array([[init_pi for j in range(img_shape[1])]for i in range(img_shape[0])]) # initializes pi for each pixel in the input image
mu = np.array([[[init_mu for k in range(K)] for j in range(img_shape[1])]for i in range(img_shape[0])]) # initializes mean of each Gaussian for each pixel in input image
sigma = np.array([[[init_sigma for k in range(K)] for j in range(img_shape[1])]for i in range(img_shape[0])]) # initializes mean of each Gaussian for each pixel in input image

# B is a bitmap that indicates howmany Gaussians are considered per pixel as background
B = np.ones(img_shape[0:2], dtype=np.int32) # initializing B as the first Gaussian for all pixels

# Initializing mean of all Gaussians for a particular pixel as the intensity value of the pixel 
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        for k in range(K):
            mu[i,j,k] = img[i,j]
alpha = init_alpha

# Defining pdf of univariate normal distribution
def univariate_normal_pdf(x,mean,std_dev):
    const = 1.0/(np.sqrt(2*np.pi*std_dev**2))
    exponent = -(x-mean)**2/(2*(std_dev**2))
    return const*np.exp(exponent)

def belongs_to_gaussian(pixel,mu,sigma):
    # If Mahalanobis distance of pixel is less than 2.5, it matches with one of the K Gaussians else not
    d = (pixel-mu)**2/sigma
    if d<2.5:
        return True
    else:
        return False
def estimate_parameters_gmm(data_dir, save_dir, frames_for_estimation):
    # data_dir = "C:\Users\BIDISHA\Desktop\ELL784_Assignment1\Frames" -> path to the folder containing the input frames where frames are stored as bmp images
    # save_dir -> path to the output folder where processed frames will be stored
    # frames_for_estimation -> no of frames required for parameter estimation
    frame_list = [] # list for storing file paths of frames
    for i in range(frames_for_estimation):
        frame_path = os.path.join(data_dir, str(i) + ".bmp")
        frame_list.append(frame_path)
    for idx, frame_path in enumerate(frame_list): 
        print("Reading frame: ", frame_path)
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2HSV)[:, :, 0] # reads the frames and converts from BGR to HSV
        frame_bgr = cv2.imread(frame_path)
        # Determining for each frame if each pixel belongs to one of the K Gaussians and setting appropriately the index of Gaussian  if match occurs
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                gauss_idx = -1
                for k in range(K):
                    if belongs_to_gaussian(frame[i, j], mu[i, j, k], sigma[i, j, k]):
                        gauss_idx = k
                        break
                # If match occurs with one of the K Gaussians, update the parameters 
                if gauss_idx != -1:
                    mu_j = mu[i, j, gauss_idx]
                    sigma_j = sigma[i, j, gauss_idx]
                    x = frame[i, j].astype(np.float32)
                    rho = alpha * univariate_normal_pdf(frame[i, j], mu_j, sigma_j)
                    for k in range(K):
                        if k == gauss_idx:
                            pi[i,j,k] = (1-alpha)*pi[i,j,k] + alpha
                        else:
                            pi[i,j,k] = (1-alpha)*pi[i,j,k]
                    mu[i,j,gauss_idx] = (1-rho)*mu_j + (rho*x)
                    sigma[i,j,gauss_idx] = (1-rho)*sigma_j + (rho*((x-mu[i,j,gauss_idx])**2))
                    # Only first B out of K Gaussians will be considered for background
                    if gauss_idx <= B[i, j]:
                        frame_bgr[i, j] = [255, 255, 255]  # Background pixels are colored white
                # If the pixel doesn't match any of the K Gaussians, the parameters of Gaussian with the least mean is replaced with the new pixel
        
                if gauss_idx == -1:
                    pi_js = [pi[i, j, k] for k in range(K)]
                    min_pi_idx = pi_js.index(min(pi_js))
                    mu[i, j, min_pi_idx] = frame[i, j]
                    sigma[i, j, min_pi_idx] = init_sigma
        # Reorder the Gaussians of each pixel for the current frame in decreasing order of pi/sigma
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                pi_vals = pi[i, j] # vector containing weights for all K Gaussians for pixel(i,j)
                sigma_vals = sigma[i, j] # vector containing sigma value of all K Gaussians for pixel(i,j)
                ordering_ratio = pi_vals / sigma_vals
                ordering = np.argsort(-ordering_ratio)
                # Ordering the Gaussians by ordering their parameters
                pi[i,j] = pi[i,j,ordering]
                mu[i,j] = mu[i,j,ordering]
                sigma[i,j] = sigma[i,j,ordering]

                sum_pi_j = 0
                for k in range(K): #Gaussians are already rearranged in descending order of pi/sigma
                    sum_pi_j += pi[i,j,k]
                    if sum_pi_j > threshold:
                        B[i,j] = k # considering the B most important Gaussians as background
                        break
        #cv2.imwrite(os.path.join(save_dir, frame_path.split("/")[-1].split(".")[0] + ".jpg"), frame_bgr) # saves the processed frame as a .jpg image
        file_name = os.path.splitext(os.path.basename(frame_path))[0] + ".jpg"
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, frame_bgr)
        print("Done!!")
if __name__ == '__main__':
    data_dir = r"C:\Users\BIDISHA\Desktop\ELL784_Assignment1\Frames"
    save_dir = r"C:\Users\BIDISHA\Desktop\ELL784_Assignment1\Processed_Frames_1DGMM"
    start_time = time.time()
    
    
    frames_for_estimation = 999  
    
    estimate_parameters_gmm(data_dir, save_dir, frames_for_estimation)

        




