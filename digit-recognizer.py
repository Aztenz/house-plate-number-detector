import cv2
import json
import os

def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def load_templates(templates_directory, num_templates):
    all_templates = []
    for digit in range(10):
        digit_templates = []
        for template_num in range(1, num_templates + 1):
            template_path = os.path.join(templates_directory, f"template-{template_num}", f"{digit}.png")
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            digit_templates.append(template)
        all_templates.append(digit_templates)
    return all_templates

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def match_template(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val

def match_digit(image, templates):
    best_match_digit = None
    best_match_score = -1

    for digit, digit_templates in enumerate(templates):
        digit_score = 0

        for template in digit_templates:
            similarity = match_template(image, template)
            digit_score += similarity

        if digit_score > best_match_score:
            best_match_digit = digit
            best_match_score = digit_score

    return best_match_digit

if __name__ == "__main__":
    test_images_directory = "test-images"
    training_dataset_path = "training.json"
    templates_directory = "digit-templates"
    num_templates_per_digit = 4

    dataset = load_dataset(training_dataset_path)
    templates = load_templates(templates_directory, num_templates_per_digit)

    total_correct_predictions = 0
    total_boxes = 0

    for entry in dataset:
        filename = entry['filename']
        if not os.path.exists(os.path.join(test_images_directory, filename)):
            continue

        boxes = entry['boxes']
        image = cv2.imread(os.path.join(test_images_directory, filename))
        preprocessed_image = preprocess_image(image)

        correct_predictions_batch = 0

        for box in boxes:
            left, top, width, height = map(int, (box['left'], box['top'], box['width'], box['height']))
            actual_digit = int(box['label'])

            roi = preprocessed_image[top:top+height, left:left+width]
            predicted_digit = match_digit(roi, templates)

            prediction_status = "Correct" if predicted_digit == actual_digit else "Incorrect"
            print(f"Filename: {filename} | Predicted: {predicted_digit} | Actual: {actual_digit} | Status: {prediction_status}")

            if predicted_digit is not None and predicted_digit == actual_digit:
                correct_predictions_batch += 1

        total_correct_predictions += correct_predictions_batch
        total_boxes += len(boxes)

    accuracy = total_correct_predictions / total_boxes
    print(f"Total Boxes: {total_boxes} | Correct Predictions: {total_correct_predictions} | Accuracy: {accuracy:.2%}")
