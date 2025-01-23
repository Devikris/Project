import os
import joblib
import cv2
import mediapipe as mp
import numpy as np

# Define the path to the 'scripts/lateralraise' directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
model_path = os.path.join(base_dir, 'mlp_model_lateralraise.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder_lateralraise.pkl')

# Load the trained model and label encoder
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
else:
    raise FileNotFoundError("One or both .pkl files are missing in the 'scripts/lateralraise' directory.")

# Get screen resolution (optional)
screen_width = 1920  # Example for 1920px width
screen_height = 1080  # Example for 1080px height

# Define the angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate cosine similarity and angle
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical issues
    angle = np.arccos(cos_angle) * (180.0 / np.pi)  # Convert radians to degrees

    return angle

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Counter and stage variables for lateral raise
counter = 0
stage = None

# Set up MediaPipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Create a named window and resize it to screen resolution
    cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mediapipe Feed', screen_width, screen_height)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image with MediaPipe Pose
        results = pose.process(image)

        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Normalize landmarks for angle calculation
            def get_coords(landmark):
                return [landmark.x, landmark.y]

            # Points for lateral raise posture
            left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            left_elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
            left_wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

            # Calculate the shoulder angle
            shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Display the angle
            cv2.putText(image, f"Shoulder Angle: {int(shoulder_angle)}", 
                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Lateral raise counting logic based on shoulder angle
            if shoulder_angle < 45:
                stage = "down"
            if shoulder_angle > 90 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Reps: {counter}")

            # Prepare feature vector for model prediction
            input_features = np.array([shoulder_angle]).reshape(1, -1)
            predicted_feedback_encoded = model.predict(input_features)[0]
            predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]

            # Display the feedback
            # cv2.putText(image, f"Feedback: {predicted_feedback}", 
            #             (10, 100), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Feedback: {predicted_feedback}", 
                        (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


        except Exception as e:
            print("Error:", e)

        # Render counter and stage
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Display rep count and stage
        cv2.putText(image, 'REPS', (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage if stage else "None", 
                    (65, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the frame
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
