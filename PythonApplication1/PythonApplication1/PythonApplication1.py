import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

gray_img = Image.open("C:/Cameraman_noise.bmp")

original_img = gray_img.copy()
third_img = gray_img.copy()
#print(img)
#plt.figure(figsize=(20,20))
#plt.imshow(img)
#plt.show()

#Function takes as arguments gray scale image and A,B,C,D to determine the contrast increase applied
#returns the filtered image
def increaseContrast(gray_img, A, B, C, D):
    width, height = gray_img.size
    gray_scale_contrast = 0;
    for i in range(0, width) :
        for j in range(0, height) :
            if(gray_img.getpixel((i, j)) < A) :
                gray_img.putpixel((i, j), gray_img.getpixel((i, j)) *(B // A))
            elif(gray_img.getpixel((i, j)) >= A and gray_img.getpixel((i, j)) < C) :
                gray_img.putpixel((i, j), ((D-B)*gray_img.getpixel((i, j)) - A*D + B*C - A*B) // (C - A))
            elif(gray_img.getpixel((i, j)) <= 255) :
                gray_img.putpixel((i, j), ((255-D)*gray_img.getpixel((i, j)) + 255*D - 255*C) // (255 - C))
    
    return gray_img

#Function takes as input the gray_img and compute the contrast using cooccurance matrix with north-south relation
#between pixels and returns the contrast value of the given image 
def computeContrast(gray_img):
    width, height = gray_img.size
    gray_scale_contrast = 0;
    for i in range(0, width) :
        for j in range(0, height-1) :
            gray_scale_contrast += (gray_img.getpixel((i, j)) - gray_img.getpixel((i, j + 1))) ** 2
    return gray_scale_contrast

def computeIntegralImage(gray_img):
    width, height = gray_img.size
    np_img = np.empty([width, height], dtype=int)

    print(np_img.shape[1])
    for j in range(0, np_img.shape[0]) :
        for i in range(0, np_img.shape[1]) :
            if (i != 0) :
                np_img[i][j] = gray_img.getpixel((i-1,j)) + gray_img.getpixel((i, j))
            else :
                np_img[i][j] = gray_img.getpixel((i, j))

    for j in range(0, np_img.shape[0]) :
        for i in range(0, np_img.shape[1]) :
            if(i == 255):
                print("Original : " + str(np_img[i][j]))
            if(j != 0) :
                np_img[i][j] = np_img[i][j-1] + np_img[i][j]
            if(i == 255) :
                print("Copy : " + str(np_img[i][j]))
            
           
                
    for j in range(0, np_img.shape[0]) :
        for i in range(0, np_img.shape[1]) :
            np_img[i][j] = (np_img[i][j] / np_img[255][255]) * 255
    
    return np_img

#filtered_img = increaseContrast(gray_img,30,20,180,230)
#filtered_img2 = increaseContrast(original_img,70,20,140,240)
#integral_img = computeIntegralImage(third_img)
#contrast = computeContrast(filtered_img)
#contrast2 = computeContrast(filtered_img2)
#print(contrast)
#print(contrast2)
np_img = computeIntegralImage(gray_img)
print(np_img[30][30])
integral_img = Image.fromarray(np_img, 'L')

#print(np_img[0])
plt.figure(figsize=(20,20))
#plt.subplot(2,2,1)
#plt.imshow(filtered_img, cmap='Greys_r')
#plt.subplot(2,2,2)
#plt.imshow(filtered_img2, cmap='Greys_r')

#uncomment these lines to draw the integral image 

#plt.plot()
#plt.imshow(integral_img, cmap='Greys_r')
#plt.show()



