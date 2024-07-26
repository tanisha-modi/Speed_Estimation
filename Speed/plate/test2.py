import cv2
import numpy as np
from scipy.signal import wiener
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
# from deblurgan.model import Generator

# # Define functions for de-blurring
def gaussian_deblur(image, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def wiener_deblur(image, mysize=None, noise=None):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    deblurred = wiener(image_gray, mysize=mysize, noise=noise)
    return np.clip(deblurred, 0, 255).astype('uint8')

# def neural_network_deblur(image_path, model_path):
#     model = Generator()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     image = Image.open(image_path)
#     preprocess = transforms.Compose([transforms.ToTensor()])
#     image_tensor = preprocess(image).unsqueeze(0)
    
#     with torch.no_grad():
#         output = model(image_tensor)
#     deblurred_image = transforms.ToPILImage()(output.squeeze(0))
    
#     return deblurred_image

# Define functions for contrast adjustment
def histogram_equalization(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(image_gray)
    return equalized

def adaptive_histogram_equalization(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image_gray)
    return equalized

def clahe_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Function to display images
def display_images(images, titles):
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(2, len(images)//2, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Main preprocessing function
def preprocess_image(image_path, model_path):
    image_path = "D:/code/Cropper/Speed/temp_vehicles/vehicle_34/frame_45.jpg"
    image = cv2.imread(image_path)
    images = [image]
    titles = ['Original Image']

    # Step 1: De-blurring
    deblurred_gaussian = gaussian_deblur(image)
    images.append(deblurred_gaussian)
    titles.append('Gaussian Deblurred')

    deblurred_wiener = wiener_deblur(image)
    images.append(deblurred_wiener)
    titles.append('Wiener Deblurred')

    # deblurred_neural = neural_network_deblur(image_path, model_path)
    # deblurred_neural = np.array(deblurred_neural)
    # images.append(deblurred_neural)
    # titles.append('Neural Network Deblurred')

    # Step 2: Contrast Adjustment
    contrast_histogram = histogram_equalization(deblurred_gaussian)
    images.append(cv2.cvtColor(contrast_histogram, cv2.COLOR_GRAY2BGR))
    titles.append('Histogram Equalization')

    contrast_adaptive_histogram = adaptive_histogram_equalization(deblurred_gaussian)
    images.append(cv2.cvtColor(contrast_adaptive_histogram, cv2.COLOR_GRAY2BGR))
    titles.append('Adaptive Histogram Equalization')

    contrast_clahe_color = clahe_color(deblurred_gaussian)
    images.append(contrast_clahe_color)
    titles.append('CLAHE Color')

    # Display all images for comparison
    display_images(images, titles)

# Call the main function
preprocess_image('blurred_image.jpg', 'deblurgan_generator.pth')
