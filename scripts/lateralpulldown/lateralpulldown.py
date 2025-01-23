import os
import joblib
import cv2
import mediapipe as mp
import numpy as np
import sys

# Check if video path is passed as an argument
if len(sys.argv) < 2:
    raise ValueError("Please provide the path to the video file as an argument.")

video_path = sys.argv[1]
print(f"Video path: {video_path}")

# Define the path to the model and label encoder
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'mlp_model_lateralpulldown.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder_lateralpulldown.pkl')

# Load the trained model and label encoder
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
else:
    raise FileNotFoundError("One or both .pkl files are missing.")

# Define the angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate cosine similarity and angle
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * (180.0 / np.pi)  # Convert radians to degrees
    return angle

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Unable to open video file: {video_path}")

# Counter and stage variables
counter = 0
stage = None
paused = False

# Set up MediaPipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image with MediaPipe Pose
            results = pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Ensure landmarks are detected
                if not results.pose_landmarks:
                    raise ValueError("Landmarks not detected.")

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Normalize landmarks for angle calculation
                def get_coords(landmark):
                    return [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]

                # Points for lateral pulldown posture
                shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

                # Calculate the elbow angle
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                print(f"Elbow Angle: {elbow_angle}")  # Debugging

                # Display the angle
                cv2.putText(image, f"Elbow Angle: {int(elbow_angle)}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Lateral pulldown counting logic
                if elbow_angle > 160 and stage != "up":  # "up" position
                    stage = "up"
                    print(f"Stage set to 'up'. Angle: {elbow_angle}")  # Debugging

                if elbow_angle < 90 and stage == "up":  # "down" position
                    counter += 1  # Increment counter when pulldown is complete
                    stage = "down"
                    print(f"Repetition counted: {counter}. Stage set to 'down'. Angle: {elbow_angle}")  # Debugging

                # Prepare feature vector for model prediction
                input_features = np.array([elbow_angle]).reshape(1, -1)
                predicted_feedback_encoded = model.predict(input_features)[0]
                predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]

                # Display the feedback
                cv2.putText(image, f"Feedback: {predicted_feedback}",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 117, 16), 2, cv2.LINE_AA)

            except Exception as e:
                print("Error:", e)

            # Render counter and stage
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Display rep count and stage
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage if stage else "None",
                        (65, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the frame
        cv2.imshow('Mediapipe Feed', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused  # Toggle pause

    cap.release()
    cv2.destroyAllWindows()
