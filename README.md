# 📄 Automatic Document Scanner using OpenCV

An automatic document scanner built using Python and OpenCV that detects a document from an input image, identifies its four corners, corrects perspective distortion, and generates a clean scanned document.

## 🚀 Features

- Detects documents from images
- Converts images to grayscale
- Reduces image noise using Gaussian Blur
- Detects edges using Canny Edge Detection
- Detects document boundaries using contours
- Identifies the four corners of the document
- Corrects perspective distortion
- Enhances the scanned document
- Saves the final scanned image

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy

## 📂 Project Structure

```text
document-scanner-opencv/
│
├── README.md
├── requirements.txt
├── main.py
├── input/
│   └── sample_document.jpg
├── output/
│   └── scanned_document.jpg
└── screenshots/
    └── result.png
