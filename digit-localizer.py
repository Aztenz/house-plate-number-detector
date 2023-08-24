import sys
import math
import cv2
import os
import json
import numpy as np
import copy
from numpy import asarray
from PIL import ImageEnhance, Image

HarshAccuracy = False


def loadDataSet(file_path: str):
    # Open the file in read-only mode
    f = open(file_path, 'r')
    # Load the contents of the file as JSON data
    data = json.load(f)
    # Return the loaded data
    return data


def EdgeDetection(imagePath, Canny):
    # Load the image from the file
    img = cv2.imread(imagePath)
    # Apply fast non-local means de-noising to the image
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Convert the de-noised image to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Define various edge detection filters as numpy arrays
    sobel_filter_vertical = np.array([[-1, 0, +1],
                                      [-2, 0, +2],
                                      [-1, 0, +1]])
    sobel_filter_horizontal = np.array([[1, 2, +1],
                                        [0, 0, 0],
                                        [-1, -2, -1]])
    laplacian_filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    laplacian_filter_strong = np.array([[-1, -1, -1],
                                        [-1, 8, -1],
                                        [-1, -1, -1]])
    hbf = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    shbf = np.array([[-1, -1, -1],
                     [-1, 9, -1],
                     [-1, -1, -1]])
    prewitt_horizontal = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
    prewitt_vertical = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])
    scharr = np.array([[-3, 0, 3],
                       [-10, 0, 10],
                       [-3, 0, 3]])
    if Canny:
        # Apply the Canny edge detection algorithm
        sharp_image_opt = cv2.filter2D(gray, -1, sobel_filter_horizontal)
        sharp_image_opt = cv2.filter2D(sharp_image_opt, -1, sobel_filter_vertical)
        sharp_image_opt = cv2.filter2D(gray, -1, prewitt_horizontal)
        sharp_image_opt = cv2.filter2D(sharp_image_opt, -1, prewitt_vertical)
        sharp_image_opt = cv2.filter2D(sharp_image_opt, -1, laplacian_filter_strong)
        sharp_image_opt = Image.fromarray(sharp_image_opt)
        enhancer = ImageEnhance.Contrast(sharp_image_opt)
        factor = 3
        sharp_image_opt = enhancer.enhance(factor)
        sharp_image_opt = asarray(sharp_image_opt)
        sharp_image_opt = cv2.Canny(gray, 50, 100)

    else:
        # Apply a series of filters for edge detection
        sharp_image_opt = cv2.filter2D(gray, -1, prewitt_horizontal)
        sharp_image_opt = cv2.filter2D(gray, -1, prewitt_vertical)
        sharp_image_opt = cv2.filter2D(gray, -1, laplacian_filter_strong)
        sharp_image_opt = Image.fromarray(sharp_image_opt)
        enhancer = ImageEnhance.Contrast(sharp_image_opt)
        factor = 3
        sharp_image_opt = enhancer.enhance(factor)
        sharp_image_opt = asarray(sharp_image_opt)
        sharp_image_opt = cv2.filter2D(gray, -1, scharr)
        sharp_image_opt = cv2.filter2D(gray, -1, shbf)

    # Display the resulting image with edges highlighted
    cv2.imshow('Edge Detection', sharp_image_opt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Estimate_digit_area(image_size):
    # Estimate the maximum and minimum sizes of the digits based on the image size
    max_digit_height = int(image_size[0] * 0.8)  # assume maximum digit height is 80% of the image height
    aspect_ratio = [0.38, 0.51, 0.54, 0.53, 0.55, 0.58, 0.53, 0.47, 0.57, 0.52]  # aspect ratio of digits 0-9
    # Assume maximum digit width is 90% of the image width, adjusted by the maximum aspect ratio
    max_digit_width = int(image_size[1] * 0.9 * max(aspect_ratio))
    # Assume minimum digit height is 10% of the image height
    min_digit_height = int(image_size[0] * 0.1)
    # Assume minimum digit width is 10% of the image width, adjusted by the minimum aspect ratio
    min_digit_width = int(image_size[1] * 0.1 * min(aspect_ratio))

    # Calculate the approximate maximum and minimum area of the digit contours based on the estimated sizes
    max_digit_area = (max_digit_height * max_digit_width)
    min_digit_area = (min_digit_height * min_digit_width)

    return min_digit_area, max_digit_area


def ROI_MSER(imagePath, automated):
    # Reads the image from the specified path
    img = cv2.imread(imagePath)

    # If automated is True, calculates the approximate region of interest (ROI)
    if automated:
        # Converts the image to grayscale and applies Canny edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp_image_opt = cv2.Canny(gray, 50, 100)

        # Calculates the minimum and maximum pixel intensity values and their corresponding locations
        (minvalue, maxvalue, minLoc, maxLoc) = cv2.minMaxLoc(sharp_image_opt)

        # Obtains the image dimensions
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

        # Calculates the starting and ending points for the bounding rectangles for the ROI
        startPointMax = ((maxLoc[0] + minLoc[0]) / 2, (maxLoc[1] + minLoc[1]) / 2)
        startPointMax2 = ((maxLoc[0] / 2 + minLoc[0]) / 2, (maxLoc[1] / 2 + minLoc[1]) / 2)
        startPointMin = ((minLoc[0] * maxLoc[0]) / 2, (minLoc[1] * maxLoc[1]) / 2)
        endPointMax = (startPointMax[0] + width / 3, startPointMax[1] + height)
        endPointMax2 = (startPointMax2[0] + width / 2, startPointMax2[1] + height)
        endPointMin = (startPointMin[0] + width / 3, startPointMin[1] + height)

        # Specifies the colors for the bounding rectangles
        colourMax = (0, 0, 0)
        colourMax2 = (0, 0, 255)
        colourMin = (255, 0, 0)

        # Draws the bounding rectangles on the original image
        blueRect = cv2.rectangle(img, (int(startPointMin[0]), int(startPointMin[1])),
                                 (int(endPointMin[0]), int(endPointMin[1])), colourMin, 2)
        blueArea = (int(startPointMin[0]), int(startPointMin[1]), int(endPointMin[0]), int(endPointMin[1]))

        blackRect = cv2.rectangle(img, (int(startPointMax[0]), int(startPointMax[1])),
                                  (int(endPointMax[0]), int(endPointMax[1])), colourMax, 2)
        redRect = cv2.rectangle(img, (int(startPointMax2[0]), int(startPointMax2[1])),
                                (int(endPointMax2[0]), int(endPointMax2[1])), colourMax2, 2)

        # Displays the original image with the bounding rectangles
        cv2.imshow("Cropped to approximate ROI", img)

    # If automated is False, allows the user to select the ROI using the mouse
    else:
        fromCenter = False
        rectangles = cv2.selectROI("Region Bounding Box", img, fromCenter)
        crop = img[int(rectangles[1]):int(rectangles[1] + rectangles[3]),
               int(rectangles[0]):int(rectangles[0] + rectangles[2])]

        # Displays the selected ROI image
        cv2.imshow("ROI Image", crop)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Corners(imagePath):
    # Read the image from the given path.
    img = cv2.imread(imagePath)

    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to the grayscale image to obtain sharp edges.
    sharp_image_opt = cv2.Canny(gray, 50, 100)

    # Convert the sharp image to a float32 format.
    gray = np.float32(sharp_image_opt)

    # Apply Harris corner detection to the sharp image with a block size of 2 and a kernel size of 3.
    # The Harris parameter k is set to 0.04.
    dst = cv2.cornerHarris(sharp_image_opt, 2, 3, 0.04)

    # Mark the detected corners in the original image by painting them red.
    # Only corners with a threshold value above 0.01 times the maximum threshold value are marked.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Display the image with detected corners in a window titled "Corners".
    cv2.imshow('Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def CannyEdge(img, showSteps=False):
    # Reduce noise using bilateral filter
    dst = cv2.bilateralFilter(img, 9, 75, 75)

    # Display intermediate step if showSteps is True
    if showSteps:
        cv2.imshow('Noise Reduction', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convert the image to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Display intermediate step if showSteps is True
    if showSteps:
        cv2.imshow('GrayScale', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Apply Canny edge detection algorithm
    thresh = cv2.Canny(gray, 50, 100)

    # Display intermediate step if showSteps is True
    if showSteps:
        cv2.imshow('Canny', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return the output image
    return thresh


def LocalizeDigits(img):
    # Find contours of the input image with external retrieval mode.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Estimate the minimum and maximum area of digit contours based on the input image size using a separate function.
    minArea, maxArea = Estimate_digit_area(img.shape)

    # Create an empty list to store final contours.
    finalContours = []

    # Iterate through all contours found earlier and keep only those that have an area between minArea and maxArea.
    # Append the bounding rectangle of each selected contour to the finalContours list.
    for contour in contours:
        area = cv2.contourArea(contour)
        if minArea < area < maxArea:
            finalContours.append(cv2.boundingRect(contour))

    # Return the list of bounding rectangles for localized digits.
    return finalContours


def getIntersectionPercentage(myOutput, realOutput):
    global HarshAccuracy

    # Read in black.png as a grayscale image and create two copies of it.
    img1Temp = cv2.imread('black.png', cv2.IMREAD_GRAYSCALE)
    img1 = copy.deepcopy(img1Temp)
    img2 = cv2.imread('black.png', cv2.IMREAD_GRAYSCALE)

    # Create an empty list to store the intersection percentages.
    allPercents = []

    # Draw rectangles on img1 at the locations specified in realOutput.
    for (x, y, w, h) in realOutput:
        cv2.rectangle(img1, (x, y), (x + w, y + h), 255, 2)

    # Draw rectangles on img2 at the locations specified in myOutput.
    for (x, y, w, h) in myOutput:
        cv2.rectangle(img2, (x, y), (x + w, y + h), 255, 3)

    # Use a bitwise AND operation to calculate the intersection of img1 and img2.
    interSection = cv2.bitwise_and(img1, img2)

    # Calculate the IoU percentage and append it to allPercents.
    if HarshAccuracy:
        allPercents.append((np.sum(interSection == 255) /
                            (np.sum(img1 == 255) + np.sum(img2 == 255) - np.sum(interSection == 255))) * 100)
    else:
        allPercents.append((np.sum(interSection == 255) /
                            (np.sum(img1 == 255))) * 100)

    # Shift the rectangles in realOutput by 5 pixels in each direction and calculate the IoU percentage again.
    for shift in [[5, 0], [-5, 0], [0, 5], [0, -5]]:
        img1 = copy.deepcopy(img1Temp)
        for (x, y, w, h) in realOutput:
            cv2.rectangle(img1, (x + shift[0], y + shift[1]), (x + w + shift[0], y + h + shift[1]), 255, 2)
        interSection = cv2.bitwise_and(img1, img2)
        if HarshAccuracy:
            allPercents.append((np.sum(interSection == 255) /
                                (np.sum(img1 == 255) + np.sum(img2 == 255) - np.sum(interSection == 255))) * 100)
        else:
            allPercents.append((np.sum(interSection == 255) /
                                (np.sum(img1 == 255))) * 100)

    # Return the maximum IoU percentage.
    return max(allPercents)


def find_encompassing_rect(rect_list):

    # Initialize the minimum and maximum x and y values as infinite and negative infinite, respectively
    min_x = float('inf')
    min_y = float('inf')
    max_x = -float('inf')
    max_y = -float('inf')
    # Initialize the maximum width and height values as negative infinite
    max_w = -float('inf')
    max_h = -float('inf')

    # Iterate through each rectangle in the provided list of rectangles
    for rect in rect_list:
        # Extract the x, y, width, and height values from the current rectangle
        (x, y, w, h) = rect
        # Update the minimum x and y values if the current x or y value is smaller than the current minimum
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        # Update maximum x and y values
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        # Update maximum width and height values
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    # Calculate and return the minimum x, minimum y, width, and height values of the encompassing rectangle
    return min_x, min_y, max_x - min_x, max_y - min_y


def Crop_To_ROI(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image using OTSU's method
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding boxes of the contours
    digit_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Threshold the grayscale image again using OTSU's method
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours of the binary image again
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add the bounding boxes of the new contours to the existing list of digit boxes
    digit_boxes.extend([cv2.boundingRect(contour) for contour in contours])

    # Find the minimum enclosing rectangle that encompasses all of the digit boxes
    (x, y, w, h) = find_encompassing_rect(digit_boxes)

    # If the minimum enclosing rectangle has a non-zero width and height, crop the image to the rectangle
    if w > 0 and h > 0:
        crop = img[y:y + h, x:x + w]
        # Display the cropped image
        cv2.imshow("Cropped Image", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def LocalizeDir(dataset, showSteps):
    percents = []
    i = 0
    total = len(os.listdir('testImages'))
    clearDir("localizedImages")
    for filename in os.listdir("testImages"):
        # Display the loading progress.
        i += 1
        loadPercent = (i/total)*100
        sys.stdout.write(f"\rLoading: [{'=' * math.floor(loadPercent/10)}{' ' * (10 - math.floor(loadPercent/10))}] "
                         f"{round(loadPercent, 1)}%")
        f = os.path.join("testImages", filename)
        imgReal = cv2.imread(f)

        # Call the 'LocalizeDigits' function to get the output.
        myOutput = LocalizeDigits(
            CannyEdge(imgReal, showSteps=showSteps))

        # Extract the expected output from the 'dataset' parameter based on the filename.
        realOutput = []
        for box in dataset[int(filename.split(".")[0]) - 1]['boxes']:
            realOutput.append((int(box['left']), int(box['top']), int(box['width']), int(box['height'])))

        # Calculate the intersection percentage and add it to the list.
        percent = getIntersectionPercentage(myOutput, realOutput)
        percents.append(percent)

        # If showSteps is True, display the localized image.
        if showSteps:
            for (x, y, w, h) in myOutput:
                cv2.rectangle(imgReal, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Localized', imgReal)
            _ = cv2.imwrite("localizedImages/localized"+filename, imgReal)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Return the list of intersection percentages.
    return percents

def clearDir(dirPath):
    try:
        files = os.listdir(dirPath)
        for file in files:
            filePath = os.path.join(dirPath, file)
            if os.path.isfile(filePath):
                os.remove(filePath)
        print("Path ["+dirPath+"] is clear!")
    except OSError:
        print("Couldn't Clear Path! relaunch application")
        exit()

# Load the dataset from the 'training.json' file using the 'loadDataSet' function.
dataSet = loadDataSet('training.json')
# Call the 'LocalizeDir' function to localize digits in the test images present in the 'testImages' folder.
percentages = LocalizeDir(dataSet, showSteps=True)
# Calculate accuracy by averaging all intersection percentages.
print(f"\n\nAccuracy is: {round(sum(percentages)/len(percentages), 1)}%")
print("Percentages:"+str(percentages))