# House Plate Number Detector

This repository contains Python scripts for detecting and recognizing house plate numbers in images. The project aims to utilize computer vision techniques to localize and recognize license plate numbers within images.

![License Plate Detection Example](images/license_plate_detection_example.png)

## Features

- License plate detection using OpenCV and contour analysis.
- Character segmentation and recognition using OCR (Optical Character Recognition) techniques.
- Accuracy evaluation and visualization of detected license plate numbers.

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- Tesseract OCR

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aztenz/house-plate-number-detector.git
   cd house-plate-number-detector

2. Install the required dependencies:
   ```bash
   pip install opencv-python

## Usage

1. Place images containing license plates in the test-images directory.
2. Run the license plate localizer script:
   ```bash
   python digit-localizer.py

Localized images will appear in localized-images directory
3. Run the license plate digit reconition script:
   ```bash
   python digit-recognizer.py

The output will be clearly visible in your terminal    
