# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image

#Function takes as arguments gray scale image and A,B,C,D to determine the contrast increase applied
#returns the filtered image
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

# Function takes as input the gray_img and compute the contrast using 
# cooccurance matrix with north-south relation
# between pixels and returns the contrast value of the given image 
def compute_contrast(gray_img):
    width, height = gray_img.size
    gray_scale_contrast = 0;
    for i in range(0, width) :
        for j in range(0, height-1) :
            gray_scale_contrast += (gray_img.getpixel((i, j)) - gray_img.getpixel((i, j + 1))) ** 2
    return gray_scale_contrast

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
# Initializations
# =============================================================================
img_filepath = "Cameraman_noise.bmp"
gray_img = Image.open(img_filepath)
org_img_array = np.array(gray_img)

# =============================================================================
# Testing only increase_contrast function
# =============================================================================

# TODO

# =============================================================================
# Testing only compute_integral_image function
# =============================================================================
int_img_array = compute_integral_image(org_img_array)
int_img = Image.fromarray(int_img_array)
int_img.convert("I").save("Camera_Int.tif")

# =============================================================================
# Testing average_filter with filter size of 3,then 5
# =============================================================================
array_3, image_3 = average_filter_using_int_image_array(int_img_array,3)
array_5, image_5 = average_filter_using_int_image_array(int_img_array,5)
image_3.save("Camera_Filt_3.jpg")
image_5.save("Camera_Filt_5.jpg")

#filtered_img = increaseContrast(gray_img,30,20,180,230)
#filtered_img2 = increaseContrast(original_img,70,20,140,240)
#integral_img = computeIntegralImage(third_img)
#contrast = computeContrast(filtered_img)
#contrast2 = computeContrast(filtered_img2)
#print(contrast)
#print(contrast2)
#print(np_img[30][30])
#integral_img = Image.fromarray(np_img, 'L')

#print(np_img[0])
#plt.figure(figsize=(20,20))
#plt.subplot(2,2,1)
#plt.imshow(filtered_img, cmap='Greys_r')
#plt.subplot(2,2,2)
#plt.imshow(filtered_img2, cmap='Greys_r')

#uncomment these lines to draw the integral image 

#plt.plot()
#plt.imshow(integral_img, cmap='Greys_r')
#plt.show()



#original_img = gray_img.copy()
#third_img = gray_img.copy()