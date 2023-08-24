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


def match_features(img1, img2, v=False):
    try:
        # Create a brute-force matcher
        brute_force_matcher = cv2.BFMatcher()

        # Extract features (key points and descriptors) from both images
        key_points_1, descriptors1 = extract_features(img1)
        key_points_2, descriptors2 = extract_features(img2)

        # Perform matching of descriptors between the two images
        matches = brute_force_matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Perform ratio test to filter out ambiguous matches
        optimizedMatches = [firstImageMatch for firstImageMatch, secondImageMatch in matches
                            if firstImageMatch.distance < 0.75 * secondImageMatch.distance]

        # Compute normalized scores and similarity sum
        similarity_sum = 0.0
        max_distance = float('-inf')
        min_distance = float('inf')
        for match in optimizedMatches:
            distance = match.distance
            similarity_sum += distance
            if distance > max_distance:
                max_distance = distance
            if distance < min_distance:
                min_distance = distance

        # Compute the average normalized score as a measure of similarity
        normalized_scores = [(max_distance - score) / (max_distance - min_distance + 0.0000001) for score in
                             (match.distance for match in optimizedMatches)]

        # Draw the matched key points on the image (if enabled)
        matched_image = cv2.drawMatches(img1, key_points_1, img2, key_points_2, optimizedMatches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if v:
            # Display the matched image (if enabled)
            cv2.imshow('Digit', matched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return similarity_sum / len(normalized_scores) if normalized_scores else 0.0
    except:
        # Return infinity if an exception occurs during the process
        return math.inf


def test_images(v=False):
    # Load the dataset file
    print("Loading DataSet File..")
    data_set = load_dataset('training.json')
    print("DataSet File Loaded!!\n")
    accuracy = []

    # Preprocess digit templates
    digit_templates = []
    for digit_filename in os.listdir("digit-templates"):
        template = os.path.join("digit-templates", digit_filename)
        digit_template = cv2.imread(template)
        digit_templates.append(digit_template)

    directory = "test-images"
    all_files = os.listdir(directory)
    total = len(all_files)

    for i, filename in enumerate(all_files, start=1):
        load_percent = (i / total) * 100
        if not v:
            sys.stdout.write(
                f"\rComputing Accuracy: [{'=' * math.floor(load_percent / 10)}{' ' * (10 - math.floor(load_percent / 10))}]"
                f"{round(load_percent, 1)}%")

        # Read the real image
        image_real = cv2.imread(os.path.join(directory, filename))
        # Get the bounding boxes for the current image
        boxes = data_set[int(filename.split(".")[0]) - 1]['boxes']
        # Normalize the real image
        normalized_image = normalize_image(image_real)

        for idx_box, box in enumerate(boxes):
            # Extract the region of interest (ROI) from the normalized image
            (x, y, w, h) = int(box['left']), int(box['top']), int(box['width']), int(box['height'])
            normalized_roi = normalized_image[y:y+h, x:x+w]

            # Calculate similarities for all digit templates
            similarities = []
            for digit_template in digit_templates:
                desired_height = h
                aspect_ratio = digit_template.shape[1] / digit_template.shape[0]
                desired_width = int(desired_height * aspect_ratio)
                resized_template = cv2.resize(digit_template, (desired_width, desired_height))
                sim = match_features(resized_template, normalized_roi, v)
                similarities.append(sim)

            # Find the best match and get the corresponding digit
            best_match_idx = similarities.index(min(similarities))
            predicted_digit = os.listdir("digit-templates")[best_match_idx].split(".")[0]

            if v:
                # Display the predicted digit on the ROI image (if enabled)
                image = cv2.cvtColor(image_real[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                aspect_ratio = image.shape[1] / image.shape[0]
                image = cv2.resize(image, (int(500 * aspect_ratio), 500))
                image = cv2.putText(image, predicted_digit, (image.shape[1] // 2, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                    3, (0, 255, 0), 5, cv2.LINE_AA)
                cv2.imshow('Digit', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(f"\nImage {filename.split('.')[0]}, Box:{idx_box}: Label = {box['label']},"
                      f" Predicted Outcome = {predicted_digit}")

            # Compute the accuracy by comparing the predicted digit with the label
            accuracy.append(predicted_digit == str(box['label']).split(".")[0])

    if not v:
        sys.stdout.write(f"\rComputing Accuracy: [{'=' * 10}] 100%")
    return sum(accuracy) / len(accuracy)


acc = test_images(v=False)
print(f"\n\nAccuracy: {round(acc * 100, 1)}%")
