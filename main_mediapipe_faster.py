import face_recognition
import cv2
import numpy as np
import os
import threading
import queue
import time
import pickle

# Optional: for faster nearest neighbor search if you have many known faces
# from sklearn.neighbors import NearestNeighbors

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

    known_face_encodings = np.array(known_face_encodings)

    # Optional: If you have a large number of known faces, use NearestNeighbors
    # global nbrs
    # if len(known_face_encodings) > 0:
    #     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(known_face_encodings)

    return known_face_encodings, known_face_names

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

    print("Initializing Camera...")
    video_capture = VideoCaptureThread(src=0, width=720, height=720, queue_size=2)
    video_capture.start()
    print("Started Video Thread...")

    # Process fewer frames to increase speed
    process_every_n_frames = 3
    frame_count = 0

    face_locations = []
    face_encodings_list = []
    face_names = []

    # Variables for "tracking"
    previous_face_locations = []
    previous_face_names = []

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            if video_capture.more():
                frame = video_capture.read()
                frame_count += 1

                # Only process every nth frame
                if frame_count % process_every_n_frames == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    results = face_detection.process(rgb_small_frame)

                    new_face_locations = []
                    new_face_names = []
                    face_encodings_list = []

                    if results.detections:
                        # Extract detected faces
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = rgb_small_frame.shape
                            x_min = int(bboxC.xmin * iw)
                            y_min = int(bboxC.ymin * ih)
                            width = int(bboxC.width * iw)
                            height = int(bboxC.height * ih)

                            top = y_min
                            right = x_min + width
                            bottom = y_min + height
                            left = x_min

                            new_face_locations.append((top, right, bottom, left))

                        # Attempt to reuse identities from previous frame if faces are stable
                        # Define a threshold for movement (in scaled coordinates)
                        movement_threshold = 20
                        recognized_indices = set()

                        for i, (top, right, bottom, left) in enumerate(new_face_locations):
                            match_found = False
                            if previous_face_locations and previous_face_names:
                                for j, (prev_top, prev_right, prev_bottom, prev_left) in enumerate(previous_face_locations):
                                    # Check if close to previous position
                                    if (abs(top - prev_top) < movement_threshold and 
                                        abs(right - prev_right) < movement_threshold and
                                        abs(bottom - prev_bottom) < movement_threshold and
                                        abs(left - prev_left) < movement_threshold):
                                        # Reuse the same identity without re-encoding
                                        new_face_names.append(previous_face_names[j])
                                        match_found = True
                                        recognized_indices.add(i)
                                        break

                            if not match_found:
                                # Need to encode this face
                                # Encode the face
                                face_encoding = face_recognition.face_encodings(rgb_small_frame, [(top, right, bottom, left)])
                                if face_encoding:
                                    face_encoding = face_encoding[0]
                                    # If you have a large dataset, use nearest neighbors
                                    # if len(known_face_encodings) > 0:
                                    #     distances, indices = nbrs.kneighbors([face_encoding])
                                    #     dist = distances[0][0]
                                    #     best_match_index = indices[0][0]
                                    #     if dist <= 0.6:
                                    #         name = known_face_names[best_match_index]
                                    #         confidence = (1 - dist) * 100
                                    #     else:
                                    #         name = "Unknown"
                                    #         confidence = 100.0
                                    # else:

                                    if len(known_face_encodings) > 0:
                                        distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                                        best_match_index = np.argmin(distances)
                                        dist = distances[best_match_index]
                                        if dist <= 0.6:
                                            name = known_face_names[best_match_index]
                                            confidence = (1 - dist) * 100
                                        else:
                                            name = "Unknown"
                                            confidence = 100.0
                                    else:
                                        # No known faces
                                        name = "Unknown"
                                        confidence = 100.0

                                    new_face_names.append((name, confidence))
                                else:
                                    # No encoding found, treat as unknown
                                    new_face_names.append(("Unknown", 100.0))

                        # Update for next iteration
                        previous_face_locations = new_face_locations
                        previous_face_names = new_face_names
                    else:
                        # No faces detected this frame
                        new_face_names = []
                        previous_face_locations = []
                        previous_face_names = []

                    face_locations = new_face_locations
                    face_names = new_face_names

                # Display results
                for ((top, right, bottom, left), (name, confidence)) in zip(face_locations, face_names):
                    # Scale back up by factor of 2 since we scaled down the image
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    label = f"{name} ({confidence:.0f}%)"
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Cleanup
        video_capture.stop()
        video_capture.join()
        cv2.destroyAllWindows()
