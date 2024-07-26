# import numpy as np
# import cv2
# import  imutils
# import sys
# import pytesseract
# import pandas as pd
# import time

# image = cv2.imread('D:/code/Cropper/Speed/car_plate.jpg')

# image = imutils.resize(image, width=500)

# cv2.imshow("Original Image", image)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #cv2.imshow("1 - Grayscale Conversion", gray)

# gray = cv2.bilateralFilter(gray, 11, 17, 17)
# #cv2.imshow("2 - Bilateral Filter", gray)

# edged = cv2.Canny(gray, 170, 200)
# #cv2.imshow("4 - Canny Edges", edged)

# (new, cnts) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
# NumberPlateCnt = None 

# count = 0
# for c in cnts:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         if len(approx) == 4:  
#             NumberPlateCnt = approx 
#             break

# # Masking the part other than the number plate
# mask = np.zeros(gray.shape,np.uint8)
# new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
# new_image = cv2.bitwise_and(image,image,mask=mask)
# cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
# cv2.imshow("Final_image",new_image)

# # Configuration for tesseract
# config = ('-l eng --oem 1 --psm 3')

# # Run tesseract OCR on image
# text = pytesseract.image_to_string(new_image, config=config)

# #Data is stored in CSV file
# raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
#         'v_number': [text]}

# df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
# df.to_csv('data.csv')

# # Print recognized text
# print(text)

# cv2.waitKey(0)



import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Read and resize the image
image = cv2.imread('D:/code/Cropper/Speed/car_plate.jpg')
if image is None:
    print("Error: Unable to read the image file.")
    sys.exit(1)
image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Detect edges
edged = cv2.Canny(gray, 170, 200)

# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)  # This ensures compatibility with different OpenCV versions

# Check if any contours were found
if not cnts:
    print("No contours found.")
    sys.exit(1)

# Sort and keep the largest 30 contours
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None

# Loop over contours to find the number plate
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx
        break

# Check if a number plate contour was found
if NumberPlateCnt is None:
    print("Number plate contour not found.")
    sys.exit(1)

# Masking the part other than the number plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
cv2.imshow("Final_image", new_image)

# Configuration for Tesseract OCR
config = ('-l eng --oem 1 --psm 3')

# Run Tesseract OCR on the image
text = pytesseract.image_to_string(new_image, config=config)

# Save data to CSV
raw_data = {'date': [time.asctime(time.localtime(time.time()))], 'v_number': [text]}
df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
df.to_csv('data.csv', index=False)

# Print recognized text
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
