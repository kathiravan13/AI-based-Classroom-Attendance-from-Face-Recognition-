# AI-based-Classroom-Attendance-from-Face-Recognition-



# AIM

To build a system that automatically marks classroom attendance using AI-based face recognition, reducing manual roll call time and improving accuracy.

# THEORY

Neural Network Model

The system primarily utilizes a Deep Learning model for face recognition, often based on a Siamese Network or Triplet Loss (e.g., FaceNet or a derivative CNN architecture like ResNet) trained for high-dimensional embedding.

Face Detection: A lightweight model (like MTCNN or Haar Cascades) locates faces in the input feed.

Feature Embedding: The recognized face is passed through the CNN to generate a unique, 128-dimensional embedding vector.

Recognition/Verification: The generated embedding is compared using Euclidean Distance against the stored embeddings of registered students. A distance below a set threshold signifies a match and marks the student as Present.

# DESIGN STEPS

STEP 1:
Enrollment Module: Capture multiple images of a student, detect the face, and generate a unique 128-D embedding vector. Store this vector alongside the student ID in the database (Firestore).

STEP 2:
Real-time Processing: Accept live video feed from a camera or an uploaded image. Process frames to detect all faces present.

STEP 3:
Recognition and Attendance Marking: For each detected face, generate the embedding and compare it against all stored embeddings in the database to identify the student. Mark the corresponding student ID as 'Present' with a timestamp.

STEP 4:
Database Integration (Firestore): Use Google Firestore to manage two main collections: StudentProfiles (for name, ID, and face embeddings) and AttendanceLogs (for daily attendance records).

STEP 5:
Web Dashboard Development: Build a teacher dashboard (using HTML/JS/Tailwind) to securely view the AttendanceLogs, correct errors, filter by date, and export data as CSV.

# PROGRAM

# Name: Kathiravan (212222230063)



# Register Number: 212222230063



This is a conceptual Python script outlining the core logic using standard libraries.
```
import cv2
import face_recognition # Assumes installation of DLib/face_recognition
import numpy as np
```
# Placeholder for Firebase/Firestore imports

# --- STEP 1: Load Enrolled Data ---
def load_known_faces(db_client):
    """Loads student embeddings and IDs from the database."""
    known_face_encodings = []
    known_face_names = []
    
    # In a real system, this would query Firestore for embeddings
    # known_face_encodings.append(np.array(db_record['embedding']))
    # known_face_names.append(db_record['student_id'])

    # Mock Data for demonstration
    mock_embedding = np.random.rand(128)
    known_face_encodings.append(mock_embedding)
    known_face_names.append("2122230063-Kathiravan")
    
    return known_face_encodings, known_face_names

# --- STEP 2 & 3: Process Live Feed ---
def process_live_attendance(known_encodings, known_names):
    video_capture = cv2.VideoCapture(0)
    attendance_marked = set()
    
    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < 0.6: # Threshold
                name = known_names[best_match_index]
                if name not in attendance_marked:
                    print(f"Marking attendance for: {name}")
                    # --- STEP 4: Database Update Placeholder ---
                    # update_firestore_attendance_log(name, "Present")
                    attendance_marked.add(name)

            # Draw a box around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('AI Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# --- Main Execution ---
```
if __name__ == "__main__":
    # db_client = initialize_firestore_client() # Placeholder
    known_encodings, known_names = load_known_faces(None)
    # process_live_attendance(known_encodings, known_names)
    print("System Ready. Run the process_live_attendance function to start camera feed.")

```



# OUTPUT

Attendance Report Snapshot

This table shows a snapshot of the attendance log for a given class on a specific date, exportable from the teacher dashboard.

# RESULT

Thus, a comprehensive plan and conceptual code for an AI-based Classroom Attendance System using face recognition, deep learning embeddings, and a web dashboard has been defined. The system is designed to automatically detect, recognize, and log student presence, fulfilling all stated project objectives.
