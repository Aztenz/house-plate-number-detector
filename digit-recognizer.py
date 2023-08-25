import math
import cv2
import os
import sys
import numpy as np
import json


def load_dataset(file_path: str):
    # Open the file in read-only mode
    f = open(file_path, 'r')
    # Load the contents of the file as JSON data
    data = json.load(f)
    # Return the loaded data
    return list(data)


def preprocess_template(template):
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    blurred_template = cv2.GaussianBlur(gray_template, (5, 5), 0)
    _, binarized_template = cv2.threshold(blurred_template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binarized_template


def normalize_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Split the grayscale image into separate planes
    rgbPlanes = cv2.split(gray)

    normalizedPlanes = []
    for plane in rgbPlanes:
        # Dilate the plane using a 7x7 kernel
        dilatedImage = cv2.dilate(plane, np.ones((7, 7), np.uint8))

        # Apply median blur with a kernel size of 21
        blurredImage = cv2.medianBlur(dilatedImage, 21)

        # Compute the absolute difference between the plane and the blurred image
        planeDifferenceImage = 255 - cv2.absdiff(plane, blurredImage)

        # Normalize the difference image to the range of 0-255
        normalizedImage = cv2.normalize(planeDifferenceImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8UC1)

        # Append the normalized plane to the list of normalized planes
        normalizedPlanes.append(normalizedImage)

    # Merge the normalized planes back into a single image
    normalizedResult = cv2.merge(normalizedPlanes)

    return normalizedResult


def extract_features(img):
    # Create a SIFT object for feature extraction
    sift = cv2.SIFT_create()

    try:
        # Try converting the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        # If conversion fails, assume the image is already grayscale
        # Detect and compute key points and descriptors using SIFT
        keyPoints, descriptors = sift.detectAndCompute(img, None)
        return keyPoints, descriptors

    # Detect and compute key points and descriptors using SIFT
    # Return the detected key points and descriptors
    keyPoints, descriptors = sift.detectAndCompute(gray, None)
    return keyPoints, descriptors


def match_template(normalized_roi, template):
    roi_height, roi_width = normalized_roi.shape[:2]
    template_height, template_width = template.shape[:2]
    result_height = roi_height - template_height + 1
    result_width = roi_width - template_width + 1

    # Calculate the normalized template
    template_mean = np.mean(template)
    template_normalized = template - template_mean
    template_norm = np.linalg.norm(template_normalized)

    # Initialize the result array
    result = np.zeros((result_height, result_width), dtype=np.float32)

    # Perform template matching using FFT-based convolution
    for y in range(result_height):
        for x in range(result_width):
            roi_patch = normalized_roi[y:y + template_height, x:x + template_width]
            roi_patch_mean = np.mean(roi_patch)
            roi_patch_normalized = roi_patch - roi_patch_mean
            roi_patch_norm = np.linalg.norm(roi_patch_normalized)

            correlation = np.sum(roi_patch_normalized * template_normalized)
            correlation /= (roi_patch_norm * template_norm)

            result[y, x] = correlation

    return result


def match_features(template, normalized_roi, v=False):
    result = match_template(normalized_roi, template)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if v:
        h, w = template.shape
        matched_image = normalized_roi.copy()
        cv2.rectangle(matched_image, max_loc, (max_loc[0] + w, max_loc[1] + h), 255, 2)
        cv2.imshow('Matching Result', matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return max_val


def test_images(v=False):
    data_set = load_dataset('training.json')
    accuracy = []

    digit_templates = []
    for digit_filename in os.listdir("digit-templates"):
        template_path = os.path.join("digit-templates", digit_filename)
        template = cv2.imread(template_path)
        preprocessed_template = preprocess_template(template)
        digit_templates.append(preprocessed_template)

    directory = "test-images"
    all_files = os.listdir(directory)
    total = len(all_files)

    for i, filename in enumerate(all_files, start=1):
        load_percent = (i / total) * 100
        if not v:
            sys.stdout.write(
                f"\rComputing Accuracy: [{'=' * math.floor(load_percent / 10)}{' ' * (10 - math.floor(load_percent / 10))}]"
                f"{round(load_percent, 1)}%")

        image_real = cv2.imread(os.path.join(directory, filename))
        boxes = data_set[int(filename.split(".")[0]) - 1]['boxes']
        normalized_image = normalize_image(image_real)

        for idx_box, box in enumerate(boxes):
            (x, y, w, h) = int(box['left']), int(box['top']), int(box['width']), int(box['height'])
            normalized_roi = normalized_image[y:y+h, x:x+w]

            similarities = []
            for digit_template in digit_templates:
                desired_height = h
                aspect_ratio = digit_template.shape[1] / digit_template.shape[0]
                desired_width = int(desired_height * aspect_ratio)
                resized_template = cv2.resize(digit_template, (desired_width, desired_height))
                sim = match_features(resized_template, normalized_roi, v)
                similarities.append(sim)

            best_match_idx = similarities.index(max(similarities))
            predicted_digit = os.listdir("digit-templates")[best_match_idx].split(".")[0]

            accuracy.append(predicted_digit == str(box['label']).split(".")[0])

    if not v:
        sys.stdout.write(f"\rComputing Accuracy: [{'=' * 10}] 100%")
    return sum(accuracy) / len(accuracy)


acc = test_images(v=False)
print(f"\n\nAccuracy: {round(acc * 100, 1)}%")
