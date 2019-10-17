# -*- coding: utf-8 -*-
#
#
# Authors: Ibram Medhat, Basem Rizk
#
# Assignment 1
# DMET901 - Computer Vision 
# The German University Of Cairo
#

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
#Function takes as arguments gray scale image and A,B,C,D to determine the contrast increase applied
#returns the filtered image
# =============================================================================

def increase_contrast(gray_img, A, B, C, D):
    width, height = gray_img.size
#    gray_scale_contrast = 0;
    for i in range(0, width) :
        for j in range(0, height) :
            if(gray_img.getpixel((i, j)) < A) :
                gray_img.putpixel((i, j), gray_img.getpixel((i, j)) *(B // A))
            elif(gray_img.getpixel((i, j)) >= A and gray_img.getpixel((i, j)) < C) :
                gray_img.putpixel((i, j), ((D-B)*gray_img.getpixel((i, j)) - A*D + B*C - A*B) // (C - A))
            elif(gray_img.getpixel((i, j)) <= 255) :
                gray_img.putpixel((i, j), ((255-D)*gray_img.getpixel((i, j)) + 255*D - 255*C) // (255 - C))
    
    return gray_img

# =============================================================================
# Function takes as input the gray_img and compute the contrast using 
# cooccurance matrix with north-south relation
# between pixels and returns the contrast value of the given image 
# =============================================================================
def compute_contrast(gray_img):
    width, height = gray_img.size
    gray_scale_contrast = 0;
    for i in range(0, width) :
        for j in range(0, height-1) :
            gray_scale_contrast += (gray_img.getpixel((i, j)) - gray_img.getpixel((i, j + 1))) ** 2
    return gray_scale_contrast

# =============================================================================
# Function takes as input the a numpy array representing an original grayscale image
# @returns the corresponding integral image array representation
# =============================================================================
def compute_integral_image(org_img_array):
#    org_img_array = np.array(gray_img)
    int_img_array = np.zeros([org_img_array.shape[0], org_img_array.shape[1]], dtype=int)

    for i in range(0, org_img_array.shape[0]) :
        for j in range(0, org_img_array.shape[1]) :
            if (j != 0 ):
                int_img_array[i][j] = int_img_array[i][j-1] + org_img_array[i][j] 
            else:
                int_img_array[i][j] = org_img_array[i, j]

    for i in range(1, org_img_array.shape[0]) :
        for j in range(0, org_img_array.shape[1]) :
            int_img_array[i][j] = int_img_array[i-1][j] + int_img_array[i][j] 
    
    return int_img_array

# =============================================================================
# Function takes as input a numpy integral image array,
# @returns a tuple containing
# (numpy array of the filtered image, filtered image as Pillow instance)
# =============================================================================
def average_filter_using_int_image_array(int_img_array, filter_size = 3): 
    print("About to apply averaging filter, using Integral Image computation")
    print(int_img_array)
        
    filtered_img_array = np.zeros([int_img_array.shape[0], int_img_array.shape[1]], dtype=int)
    filter_padding = filter_size//2
    for i in range(filter_padding + 1, int_img_array.shape[0] - filter_padding - 1):
        for j in range(filter_padding + 1, int_img_array.shape[1] - filter_padding - 1):
            
            cummulative_diff =\
                    int_img_array[i+filter_padding][j+filter_padding] +\
                    int_img_array[i-filter_padding-1][j-filter_padding-1] -\
                    int_img_array[i+filter_padding][j-filter_padding-1] -\
                    int_img_array[i-filter_padding-1][j+filter_padding]
            cummulative_diff = cummulative_diff / (filter_size**2)
#            print(cummulative_diff)
            filtered_img_array[i][j] = cummulative_diff
                
    filtered_img = Image.fromarray(filtered_img_array)
    filtered_img = filtered_img.convert("L")
    return filtered_img_array, filtered_img

# =============================================================================
#                           Problem 2 :: C
# ============================================================================= 
# The Function average_filter_using_int_image_array uses the integral image
# representation of an image to compute a filtered version of it.
#
# The advantage of using the integral image for average filtering here compared
# to traditional convolution implementation of average filtering is :
# 
# that it does not need to add up all the values of pixels lying underneath
# the area of the filter; however, takes advantage of the cummulation of values
# produced already in the integral image representation
# ; hence, less computation.
#
# =============================================================================

# =============================================================================
# Initializations
# =============================================================================
img_filepath = "C:/Cameraman_noise.bmp"
original_img = Image.open("C:/Ocean.bmp")
gray_img = Image.open(img_filepath)
org_img_array = np.array(gray_img)

# =============================================================================
# Plotting increase_contrast on first set of values and printing its contrast value
# =============================================================================

contrast_image_1 = increase_contrast(original_img,30,20,180,230)
contrast_value_1 = compute_contrast(contrast_image_1)
print("First Image Contrast is : " + str(contrast_value_1))
contrast_image_1.save("Ocean_a.bmp")

# =============================================================================
# Plotting increase_contrast on second set of values and printing its contrast value
# =============================================================================

contrast_image_2 = increase_contrast(original_img,70,20,140,240)
contrast_value_2 = compute_contrast(contrast_image_2)
print("Second Image Contrast is : " + str(contrast_value_2))
contrast_image_2.save("Ocean_b.bmp")

# =============================================================================
# Testing only compute_integral_image function
# =============================================================================
int_img_array = compute_integral_image(org_img_array)
integral_fig = plt.figure(figsize=(20,20))
integral_fig.suptitle('Integral Image')
plt.plot()
plt.imshow(int_img_array, cmap='Greys_r')
plt.show()
#After plotting the image we save it as jpg

# =============================================================================
# Testing average_filter with filter size of 3,then 5
# =============================================================================
array_3, image_3 = average_filter_using_int_image_array(int_img_array,3)
array_5, image_5 = average_filter_using_int_image_array(int_img_array,5)
image_3.save("Camera_Filt_3.jpg")
image_5.save("Camera_Filt_5.jpg")









