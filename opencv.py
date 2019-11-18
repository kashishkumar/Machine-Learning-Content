import cv2
import numpy as np
"""Basic Operations on Images"""
img=cv2.imread('wallpaper.jpg') # BGR Values, numpy array (3D) with 3 channels
import matplotlib.pyplot as plt
plt.imshow(img)
img[1,1] # Pixel at 1,1
img.item(1,1,1) # Method to access single item
img.itemset((1,1,1),100) # Modify the item
img.shape # 3D tensor
img.size  # Total number of values
img.dtype # Important for debugging
type(img)
# Region of Images help us move from larger to smaller while searching to make it faster
plt.imshow(img[100:2000,200:2500])
# Can be used to copy a part of image as well
b,g,r=cv2.split(img) # Split channels - expensive operation
img_rgb=cv2.merge((r,g,b)) # Merge channels
plt.imshow(img_rgb) # 
replicate=cv2.copyMakeBorder(img_rgb,100,100,100,100,cv2.BORDER_REPLICATE) # Padding or adding borders, different options available
plt.imshow(replicate)
"""Arithmetic Operations on Images"""
x=np.uint8([250])
y=np.uint8([10])
cv2.add(x,y)
img1=cv2.imread('img1.jpg')
img2=cv2.imread('img2.jpg')
img_add=cv2.addWeighted(img1,0.7,img2,0.3,0.1)
plt.imshow(img_add)
plt.imshow(img1)
"""Image Processing using OpenCV"""
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # Converting to RGB
plt.imshow(img_rgb)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # Converting to HSV
plt.imshow(img_hsv)
green=np.uint8([[[0,255,0]]])
green_hsv=cv2.cvtColor(green,cv2.COLOR_BGR2HSV) # Converting to HSV
plt.imshow(green_hsv)
"""Image Transformation"""
#Scaling
res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) 
plt.imshow(res)
#Translation
M=np.float32([[1,0,10],[0,1,5]]) # Transformation Matrix for translation of (10,5)
dst=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
plt.imshow(dst)
#Rotation
M=cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),90,1) # Transformation Matrix
dst_rot=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
plt.imshow(dst_rot)
# Affine Transformation
#All parallel lines remain parallel

"""Image Filtering"""
kernel = np.ones((5,5),np.float32)/25
dst_convolved=cv2.filter2D(img,-1,kernel)
plt.imshow(dst_convolved)
plt.imshow(img)

blur=cv2.blur(img,(7,7)) # Using box filter
plt.imshow(blur)

blur_gaussian=cv2.GaussianBlur(img,(5,5),0)
plt.imshow(blur_gaussian)

# Averaging, Gaussian Filtering, Median Filtering and Bilateral Filtering

"""Morphological Transformations"""
#Black and white image would be better
#Erosion - Erodes away the boundaries of foreground objects
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
plt.imshow(erosion)
#Dilation -  increases the white region in the image or size of foreground object increases.
dilation = cv2.dilate(img,kernel,iterations = 1)
plt.imshow(dilation)
#Opening - Erosion followed by dilation for removing noise
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
plt.imshow(opening)
plt.imshow(img)
# Closing - Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing)
# Morphological Gradient - Difference between dilation and erosion of image (outline of image)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.imshow(gradient)

cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))