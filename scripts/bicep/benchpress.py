import cv2
import mediapipe as mp
import numpy as np

# Get screen resolution (optional, just to resize the window)
screen_width = 1920  # Example for 1920px width (you can replace this dynamically)
screen_height = 1080  # Example for 1080px height (you can replace this dynamically)

# Define the angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Counter and stage variables for squats
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

            # Hip, knee, and ankle landmarks for squat angle calculation
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate the angle between hip, knee, and ankle
            angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Display the angle on the screen
            cv2.putText(image, str(int(angle)), 
                        tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Squat counting logic
            if angle > 160:
                stage = "up"
            if angle < 90 and stage == "up":
                stage = "down"
                counter += 1
                print(f"Reps: {counter}")

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
        cv2.putText(image, stage, 
                    (60, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Mediapipe Feed', image)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
