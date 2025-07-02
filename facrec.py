import face_recognition
import cv2
import os
import numpy as np

# Path to known face images
KNOWN_FACES_DIR = "known_faces"

# Initialize known encodings and names
known_face_encodings = []
known_face_names = []

# Load and encode known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            print(f"[INFO] Loaded face for: {name}")
        else:
            print(f"[WARNING] No face found in {filename}")

# Start webcam
video_capture = cv2.VideoCapture(0)

print("🎥 Real-Time Face Recognition started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Resize frame and convert color
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale back face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)

    # Display result
    cv2.imshow('Face Recognition', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
