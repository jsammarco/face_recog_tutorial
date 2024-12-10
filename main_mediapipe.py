import face_recognition
import cv2
import numpy as np
import os
import threading
import queue
import time
import pickle

# Import Mediapipe
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []
    print("Loading encodings for faces...")

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(directory, filename)
            name, _ = os.path.splitext(filename)
            pkl_path = os.path.join(directory, f"{name}.pkl")

            if os.path.exists(pkl_path):
                # Load encoding from pickle file
                try:
                    with open(pkl_path, 'rb') as pkl_file:
                        encoding = pickle.load(pkl_file)
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                        print(f"Loaded encoding from {pkl_path}")
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")
                    # If loading fails, proceed to generate encoding
            else:
                # Generate encoding and save to pickle
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        encoding = face_encodings[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(name)
                        print(f"Generated and saved encoding for {image_path}")

                        # Save the encoding to a pickle file
                        with open(pkl_path, 'wb') as pkl_file:
                            pickle.dump(encoding, pkl_file)
                    else:
                        print(f"No faces found in {image_path}. Skipping.")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    # Convert to NumPy array for faster computations
    known_face_encodings = np.array(known_face_encodings)
    return known_face_encodings, known_face_names

# Thread class for video capture
class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0, width=640, height=480, queue_size=2):
        super().__init__()
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.capture.read()
                if not ret:
                    self.stop()
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.015)  # Prevent busy waiting

    def read(self):
        return self.queue.get()

    def more(self):
        return not self.queue.empty()

    def stop(self):
        self.stopped = True
        self.capture.release()

if __name__ == "__main__":
    # Load known faces
    known_faces_dir = "known_faces"
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    # Initialize video capture thread
    print("Initializing Camera...")
    video_capture = VideoCaptureThread(src=0, width=720, height=720, queue_size=2)
    video_capture.start()
    print("Started Video Thread...")

    process_every_n_frames = 2
    frame_count = 0

    # Initialize variables for multi-threading
    face_locations = []
    face_encodings_list = []
    face_names = []

    # Initialize Mediapipe face detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            if video_capture.more():
                frame = video_capture.read()
                frame_count += 1

                # Only process every n-th frame to save time
                if frame_count % process_every_n_frames == 0:
                    # Resize frame to 1/2 size for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    # Convert the image from BGR (OpenCV) to RGB
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # Use Mediapipe to detect faces
                    results = face_detection.process(rgb_small_frame)

                    face_locations = []
                    face_names = []
                    face_encodings_list = []

                    if results.detections:
                        for detection in results.detections:
                            # Extract bounding box
                            # Mediapipe returns normalized box coordinates
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = rgb_small_frame.shape
                            x_min = int(bboxC.xmin * iw)
                            y_min = int(bboxC.ymin * ih)
                            width = int(bboxC.width * iw)
                            height = int(bboxC.height * ih)

                            # Convert to face_recognition format: (top, right, bottom, left)
                            top = y_min
                            right = x_min + width
                            bottom = y_min + height
                            left = x_min

                            # We now have the face location in the small_frame coordinates
                            # Append to face_locations list
                            # NOTE: face_recognition expects these coordinates in order: top, right, bottom, left
                            face_locations.append((top, right, bottom, left))

                        # Get face encodings from these locations
                        # face_recognition.face_encodings requires the original RGB image and the face locations
                        face_encodings_list = face_recognition.face_encodings(rgb_small_frame, face_locations)

                        # Match faces
                        for face_encoding in face_encodings_list:
                            distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1) if len(known_face_encodings) > 0 else np.array([])
                            name = "Unknown"
                            confidence = 1.0  # Default confidence

                            if len(distances) > 0:
                                best_match_index = np.argmin(distances)
                                if distances[best_match_index] <= 0.6:
                                    name = known_face_names[best_match_index]
                                    confidence = (1 - distances[best_match_index]) * 100

                            face_names.append((name, confidence))
                    else:
                        # No faces detected
                        face_names = []

                # Display the results
                for ((top, right, bottom, left), (name, confidence)) in zip(face_locations, face_names):
                    # Scale back up face locations since the frame was scaled
                    # We scaled frame by 0.5, so we multiply by 2
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw the label
                    label = f"{name} ({confidence:.0f}%)"
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                # Show the frame
                cv2.imshow("Video", frame)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Stop the video capture thread and close windows
        video_capture.stop()
        video_capture.join()
        cv2.destroyAllWindows()
