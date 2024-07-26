# # from PIL import Image
# from PIL import Image, ImageEnhance

# def upscale_image(input_path, output_path, scale_factor):
#     with Image.open(input_path) as img:
#         new_size = tuple(int(dim * scale_factor) for dim in img.size)
#         resized_img = img.resize(new_size, Image.LANCZOS)
#         resized_img.save(output_path)



# def adjust_contrast(input_path, output_path, factor):
#     with Image.open(input_path) as img:
#         enhancer = ImageEnhance.Contrast(img)
#         enhanced = enhancer.enhance(factor)
#         enhanced.save(output_path)


input_path = "D:/code/Cropper/Speed/violations/violation_car_12_20240629_145810.jpg"
# output_path = "D:/code/Cropper/Speed/enhance/violation_car_12_20240629_145810.jpg"
# contrast_output_path = "D:/code/Cropper/Speed/enhance/contrast1_violation_car_12_20240629_145810.jpg"

# upscale_image(input_path, output_path, 2.0)
# adjust_contrast(input_path, contrast_output_path, 1.5)


#Import the necessary libraries 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

# Load the image 
image = cv2.imread(input_path) 

#Plot the original image 
plt.subplot(1, 2, 1) 
plt.title("Original") 
plt.imshow(image) 

# Create the sharpening kernel 
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

# Sharpen the image 
sharpened_image = cv2.filter2D(image, -1, kernel) 

#Save the image 
cv2.imwrite('D:/code/Cropper/Speed/enhance/sharpened_image.jpg', sharpened_image) 

#Plot the sharpened image 
plt.subplot(1, 2, 2) 
plt.title("Sharpening") 
plt.imshow(sharpened_image) 
plt.show()
