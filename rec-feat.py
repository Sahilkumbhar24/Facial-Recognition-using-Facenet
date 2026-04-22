import cv2
import numpy as np
import os
import sys
import glob
import face_recognition

# ---- Configuration ----
RECOGNITION_THRESHOLD = 0.5  # Lower = stricter matching (typical: 0.4-0.6)

def load_database(images_dir="images"):
    database = {}
    for entry in glob.glob(os.path.join(images_dir, "*")):
        if os.path.isdir(entry):
            identity = os.path.basename(entry)
            image_files = glob.glob(os.path.join(entry, "*"))
            encodings = []
            for img_file in image_files:
                if os.path.isfile(img_file):
                    print(f"  Encoding {identity}: {os.path.basename(img_file)}", flush=True)
                    try:
                        img = face_recognition.load_image_file(img_file)
                        face_encs = face_recognition.face_encodings(img)
                        if len(face_encs) > 0:
                            encodings.append(face_encs[0])
                        else:
                            print(f"    Warning: No face found in {img_file}", flush=True)
                    except Exception as e:
                        print(f"    Error processing {img_file}: {e}", flush=True)
            if len(encodings) > 0:
                avg_encoding = np.mean(encodings, axis=0)
                database[identity] = avg_encoding
                print(f"  -> {identity}: averaged {len(encodings)} images", flush=True)
        else:
            identity = os.path.splitext(os.path.basename(entry))[0]
            print(f"  Encoding {identity} (single image)", flush=True)
            try:
                img = face_recognition.load_image_file(entry)
                face_encs = face_recognition.face_encodings(img)
                if len(face_encs) > 0:
                    database[identity] = face_encs[0]
            except Exception as e:
                print(f"    Error processing {entry}: {e}", flush=True)

    print(f"\nDatabase loaded: {list(database.keys())}", flush=True)
    return database


def recognize():
    print("=" * 50, flush=True)
    print("  Structural Face Recognition System", flush=True)
    print("=" * 50, flush=True)
    
    database = load_database()
    if not database:
        print("ERROR: Database empty.", flush=True)
        return

    known_names = list(database.keys())
    known_encodings = list(database.values())

    print("\nAttempting to open webcam...", flush=True)
    # Try multiple backends and indices if one fails
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Standard backend failed, trying DirectShow...", flush=True)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam index 0.", flush=True)
        return

    print("Webcam started. Press 'q' in the window to quit.", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame", flush=True)
            break

        # Process at 1/4 size for much better speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all faces and their encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Compare
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255) # Red for unknown
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < RECOGNITION_THRESHOLD:
                    name = known_names[best_match_index]
                    color = (0, 255, 0) # Green for known

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
