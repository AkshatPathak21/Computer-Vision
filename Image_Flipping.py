import cv2
from matplotlib import pyplot as plt

#Reading the image and converting it to array form
img = cv2.imread("C:\Python Program\Computer Vision\Jett.jpg")
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
name = "Jett"
plt.title(name)
(plt.imshow(RGB_img[::]))
plt.show()

# #Horizontal Flip
HF = cv2.flip(RGB_img, 1)
name = "Horizontal Flip"
plt.title(name)
plt.imshow(HF)
plt.show()

#Vertical Flip
VF = cv2.flip(RGB_img, 0)
name = "Vertical Flip"
plt.title(name)
plt.imshow(VF)
plt.show()

# #Vertical Flip after horizontal flip
VF_HF = cv2.flip(HF, 0)
name = "Vertical flip after horizontal flip"
plt.title(name)
plt.imshow(VF_HF)
plt.show()

# # Horizontal flip after vertical flip\
HF_VF = cv2.flip(VF,1)
name = "Horizontal flip after vertical flip"
plt.title(name)
plt.imshow(HF_VF)
plt.show()

# #Horizontaly and verticaly flip
HVF = cv2.flip(RGB_img, -1)
name = "Horizontally and Vertically Flipped"
plt.title(name)
plt.imshow(HVF)
plt.show()



# Horizontal flip
horizontal_flip = cv2.flip(img, 1)

# Vertical flip
vertical_flip = cv2.flip(img, 0)

# Display the original image and the flipped images
cv2.imshow('Original', img)
cv2.imshow('Horizontal Flip', horizontal_flip)
cv2.imshow('Vertical Flip', vertical_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()
