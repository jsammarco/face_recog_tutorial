import face_recognition
import cv2
import numpy as np
import os
import threading
import queue
import time
import pickle

# Step 1: Encode the known faces with caching
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

# Main function to use webcam
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
    face_encodings = []
    face_names = []

    while True:
        if video_capture.more():
            frame = video_capture.read()
            frame_count += 1

            # Only process every n-th frame to save time
            if frame_count % process_every_n_frames == 0:
                # Resize frame to 1/2 size for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                # Convert the image from BGR (OpenCV) to RGB (face_recognition)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Detect all faces and their encodings in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # Compare the detected face with known faces
                    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                    best_match_index = np.argmin(distances)

                    name = "Unknown"
                    confidence = 1.0  # Default confidence for unknown faces

                    if distances[best_match_index] <= 0.6:  # 0.6 is a common threshold
                        name = known_face_names[best_match_index]
                        confidence = (1 - distances[best_match_index]) * 100  # Convert to percentage

                    face_names.append((name, confidence))

            # Display the results
            for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw the label with the name and confidence score
                label = f"{name} ({confidence:.0f}%)"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Display the resulting frame
            cv2.imshow("Video", frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Stop the video capture thread and close windows
    video_capture.stop()
    video_capture.join()
    cv2.destroyAllWindows()
