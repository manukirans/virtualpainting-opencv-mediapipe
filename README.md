# Virtual Painting

## Description

The Virtual Painting Application is an interactive program that allows users to draw and paint in real-time using hand gestures captured through a webcam. The application leverages the power of OpenCV for video processing, MediaPipe for hand tracking, and the Savitzky-Golay filter for smoothing the drawing. Users can select different colors and draw on a virtual canvas by simply moving their hands in front of the camera.

## Key Features

- Real-time hand detection and tracking.
- Drawing on a virtual canvas using the index finger.
- Smoothening of drawn lines using the Savitzky-Golay filter.
- Color selection through a user interface with header images.
- Eraser functionality.

## Dependencies

- OpenCV
- MediaPipe
- NumPy
- SciPy

## Installation

1. **Install the required libraries:**
```sh
pip install opencv-contrib-python
pip install mediapipe

```
2. **Header Images**
- Ensure you have the header images stored in a folder named "Headers" within the project directory. Each image should represent a different color option for the drawing. 
 

  
