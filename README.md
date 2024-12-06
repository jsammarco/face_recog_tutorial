# Face Recognition Tutorial

This repository contains a Python program for real-time face recognition using a webcam.
It utilizes the `face_recognition` library for facial feature detection and comparison,
and `OpenCV` for video processing. Encoded faces are cached in pickle files for faster recognition.

## Features

- Load and cache known face encodings from images.
- Perform real-time face detection and recognition from a webcam.
- Display face recognition results with confidence percentages.
- Multi-threaded video capture for improved performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jsammarco/face_recog_tutorial.git
   cd face_recog_tutorial
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure that you have the following installed:
   - Python 3.6 or later
   - `face_recognition`
   - `opencv-python`
   - `numpy`

3. Set up the `known_faces` directory:
   - Place images of known faces in the `known_faces` directory.
   - Each image file should be named after the person in the photo (e.g., `John_Doe.jpg`).

## Usage

Run the program to start the face recognition system:
```bash
python face_recognition.py
```

### Keyboard Shortcuts
- Press `q` to quit the program.

### Notes
- The program processes every second frame by default for improved performance.
- Face encodings are automatically saved as pickle files (`.pkl`) in the same directory as the source images.

## Directory Structure

```
face_recog_tutorial/
├── face_recognition.py   # Main program file
├── known_faces/          # Directory for storing known face images
├── requirements.txt      # List of required Python packages
```

## Requirements

- Python 3.6+
- OpenCV
- Face Recognition Library
- Numpy

## Troubleshooting

- If the camera does not initialize, ensure it is connected and accessible.
- If the program cannot find faces in the images, check that the images are clear and contain a single face.

## License

This project is open source and available under the MIT License.
