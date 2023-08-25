import sys
import math
import cv2
import os
import json
import numpy as np
import copy

HarshAccuracy = False


def load_dataset(file_path: str):
    # Open the file in read-only mode
    f = open(file_path, 'r')
    # Load the contents of the file as JSON data
    data = json.load(f)
    # Return the loaded data
    return data


def estimate_digit_area(image_size):
    # Estimate the maximum and minimum sizes of the digits based on the image size
    # assume maximum digit height is 80% of the image height
    max_digit_height = int(image_size[0] * 0.8)
    aspect_ratio = [0.38, 0.51, 0.54, 0.53, 0.55, 0.58,
                    0.53, 0.47, 0.57, 0.52]  # aspect ratio of digits 0-9
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


def canny_edge(img, showSteps=False):
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


def localize_digits(img):
    # Find contours of the input image with external retrieval mode.
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Estimate the minimum and maximum area of digit contours based on the input image size using a separate function.
    minArea, maxArea = estimate_digit_area(img.shape)

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


def get_intersection_percentage(myOutput, realOutput):
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
            cv2.rectangle(
                img1, (x + shift[0], y + shift[1]), (x + w + shift[0], y + h + shift[1]), 255, 2)
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


def localize_dir(dataset, showSteps):
    percents = []
    i = 0
    total = len(os.listdir('test-images'))
    clear_dir("localized-images")
    for filename in os.listdir("test-images"):
        # Display the loading progress.
        i += 1
        loadPercent = (i/total)*100
        sys.stdout.write(f"\rLoading: [{'=' * math.floor(loadPercent/10)}{' ' * (10 - math.floor(loadPercent/10))}] "
                         f"{round(loadPercent, 1)}%")
        f = os.path.join("test-images", filename)
        imgReal = cv2.imread(f)

        # Call the 'localize_digits' function to get the output.
        myOutput = localize_digits(
            canny_edge(imgReal, showSteps=showSteps))

        # Extract the expected output from the 'dataset' parameter based on the filename.
        realOutput = []
        for box in dataset[int(filename.split(".")[0]) - 1]['boxes']:
            realOutput.append((int(box['left']), int(
                box['top']), int(box['width']), int(box['height'])))

        # Calculate the intersection percentage and add it to the list.
        percent = get_intersection_percentage(myOutput, realOutput)
        percents.append(percent)

        # If showSteps is True, display the localized image.
        for (x, y, w, h) in myOutput:
            cv2.rectangle(imgReal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _ = cv2.imwrite("localized-images/"+filename, imgReal)
        if showSteps:
            cv2.imshow('Localized', imgReal)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Return the list of intersection percentages.
    return percents


def clear_dir(dirPath):
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


# Load the dataset from the 'training.json' file using the 'load_dataset' function.
dataSet = load_dataset('training.json')
# Call the 'localize_dir' function to localize digits in the test images present in the 'test-images' folder.
percentages = localize_dir(dataSet, showSteps=True)
# Calculate accuracy by averaging all intersection percentages.
print(f"\n\nAccuracy is: {round(sum(percentages)/len(percentages), 1)}%")
print("Percentages:"+str(percentages))
