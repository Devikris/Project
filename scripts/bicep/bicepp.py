# # %%
# pip install mediapipe opencv-python

# # %%
# import cv2
# import mediapipe as mp
# import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # %%
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imshow('Mediapipe Feed', frame)
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
        
# cap.release()
# cv2.destroyAllWindows()

# # %%
# cap = cv2.VideoCapture(0)
# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = pose.process(image)
    
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                  )               
        
#         cv2.imshow('Mediapipe Feed', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # %%
# mp_pose.POSE_CONNECTIONS

# # %%
# results.pose_landmarks

# # %%
# mp_drawing.DrawingSpec??

# # %%
# cap = cv2.VideoCapture(0)
# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = pose.process(image)
    
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         try:
#             landmarks=results.pose_landmarks.landmark
#             print(landmarks)
#         except:
#             pass
        
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                  )               
        
#         cv2.imshow('Mediapipe Feed', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # %%
# len(landmarks)

# # %%
# landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x

# # %%
# landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

# # %%
# landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

# # %%
# def calculate_angles(a,b,c):
#     a=np.array(a)
#     b=np.array(b)
#     c=np.array(c)
#     radians=np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
#     angle=np.abs(radians*180.0/np.pi)
#     if angle>180.0:
#         angle=360-angle
#     return angle

# # %%
# shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

# # %%
# shoulder,elbow,wrist

# # %%
# calculate_angles(shoulder,elbow,wrist)

# # %%
# tuple(np.multiply(elbow, [640,480]).astype(int))

# # %%
# import cv2
# import mediapipe as mp
# import numpy as np

# # Define the angle calculation function
# def calculate_angle(a, b, c):
#     a = np.array(a)  # First point
#     b = np.array(b)  # Midpoint
#     c = np.array(c)  # End point
    
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle

# # Setup MediaPipe
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# # Set up MediaPipe Pose instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False

#         # Make detection
#         results = pose.process(image)

#         # Recolor back to BGR for rendering
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark

#             # Left side coordinates
#             left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
#                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
#                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
#                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#             # Right side coordinates
#             right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
#                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#             right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
#                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#             right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
#                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

#             # Calculate angles
#             left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
#             right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

#             # Visualize left elbow angle
#             cv2.putText(image, str(int(left_angle)), 
#                         tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
#             # Visualize right elbow angle
#             cv2.putText(image, str(int(right_angle)), 
#                         tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#             # For debugging
#             print("Left Angle:", left_angle, "Right Angle:", right_angle)

#         except Exception as e:
#             print("Error:", e)

#         # Render pose landmarks on the image
#         mp_drawing.draw_landmarks(
#             image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
#             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#         )

#         # Display the image
#         cv2.imshow('Mediapipe Feed', image)

#         # Press 'q' to exit
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# # %%
# cap = cv2.VideoCapture(0)

# # Curl counter variables
# counter = 0 
# stage = None

# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
        
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = pose.process(image)
    
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         # Extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark
            
#             # Get coordinates
#             shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
#             # Calculate angle
#             angle = calculate_angle(shoulder, elbow, wrist)
            
#             # Visualize angle
#             cv2.putText(image, str(angle), 
#                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                                 )
            
#             # Curl counter logic
#             if angle > 160:
#                 stage = "down"
#             if angle < 30 and stage =='down':
#                 stage="up"
#                 counter +=1
#                 print(counter)
                       
#         except:
#             pass
        
#         # Render curl counter
#         # Setup status box
#         cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
#         # Rep data
#         cv2.putText(image, 'REPS', (15,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#         cv2.putText(image, str(counter), 
#                     (10,60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
#         # Stage data
#         cv2.putText(image, 'STAGE', (65,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#         cv2.putText(image, stage, 
#                     (60,60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                  )               
        
#         cv2.imshow('Mediapipe Feed', image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # %%
#------------------------------------------------------------------------------------------------------------------------------
# import os
# import joblib
# import cv2
# import mediapipe as mp
# import numpy as np

# # Define the path to the 'scripts/bicep' directory
# base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
# model_path = os.path.join(base_dir, 'mlp_feedback_model.pkl')
# label_encoder_path = os.path.join(base_dir, 'feedback_label_encoder.pkl')

# # Load the trained model and label encoder
# if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#     model = joblib.load(model_path)
#     label_encoder = joblib.load(label_encoder_path)
# else:
#     raise FileNotFoundError("One or both .pkl files are missing in the 'scripts/bicep' directory.")

# # Get screen resolution (optional)
# screen_width = 1920  # Example for 1920px width
# screen_height = 1080  # Example for 1080px height

# # Define the angle calculation function
# def calculate_angle(a, b, c):
#     a = np.array(a)  # First point
#     b = np.array(b)  # Midpoint
#     c = np.array(c)  # End point

#     # Calculate vectors
#     ba = a - b
#     bc = c - b

#     # Calculate cosine similarity and angle
#     cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical issues
#     angle = np.arccos(cos_angle) * (180.0 / np.pi)  # Convert radians to degrees

#     return angle

# # Initialize MediaPipe Pose and Drawing modules
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Counter and stage variables for bicep curls
# counter = 0
# stage = None

# # Set up MediaPipe Pose instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     # Create a named window and resize it to screen resolution
#     cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Mediapipe Feed', screen_width, screen_height)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Flip the frame horizontally for a mirror effect
#         frame = cv2.flip(frame, 1)

#         # Convert the image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False

#         # Process the image with MediaPipe Pose
#         results = pose.process(image)

#         # Convert back to BGR for rendering
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         try:
#             # Extract landmarks
#             landmarks = results.pose_landmarks.landmark

#             # Normalize landmarks for angle calculation
#             def get_coords(landmark):
#                 return [landmark.x, landmark.y]

#             left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
#             left_elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
#             left_wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

#             # Calculate the angle of the left elbow
#             angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

#             # Display the angle
#             cv2.putText(image, f"Angle: {int(angle)}", 
#                         tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#             # Curl counting logic
#             if angle > 160:
#                 stage = "down"
#             if angle < 30 and stage == "down":
#                 stage = "up"
#                 counter += 1
#                 print(f"Reps: {counter}")

#             # Prepare feature vector for model prediction
#             input_features = np.array([angle]).reshape(1, -1)
#             predicted_feedback_encoded = model.predict(input_features)[0]
#             predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]

#             # Display the feedback
#             # cv2.putText(image, f"Feedback: {predicted_feedback}", 
#             #             (10, 100), 
#             #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(image, f"Feedback: {predicted_feedback}", 
#                         (30, 100), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


#         except Exception as e:
#             print("Error:", e)

#         # Render counter and stage
#         cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

#         # Display rep count and stage
#         cv2.putText(image, 'REPS', (15, 12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(image, str(counter), 
#                     (10, 60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#         cv2.putText(image, 'STAGE', (65, 12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(image, stage if stage else 'None', 
#                     (60, 60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#         # Draw pose landmarks
#         # mp_drawing.draw_landmarks(
#         #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#         #     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
#         #     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#         # )

#         # Display the frame
#         cv2.imshow('Mediapipe Feed', image)

#         # Exit on pressing 'q'
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
# -----------------------------------------code with pause old code
import os
import joblib
import cv2
import mediapipe as mp
import numpy as np

# Define the path to the 'scripts/bicep' directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
model_path = os.path.join(base_dir, 'mlp_feedback_model.pkl')
label_encoder_path = os.path.join(base_dir, 'feedback_label_encoder.pkl')

# Load the trained model and label encoder
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
else:
    raise FileNotFoundError("One or both .pkl files are missing in the 'scripts/bicep' directory.")

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

# Counter and stage variables for bicep curls
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
                return [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]

            left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            left_elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
            left_wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

            # Calculate the angle of the left elbow
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Display the angle
            cv2.putText(image, f"Angle: {int(angle)}", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counting logic
            if angle > 160:  # Arm fully extended
                stage = "down"
            if angle < 30 and stage == "down":  # Arm fully curled
                stage = "up"
                counter += 1
                print(f"Reps: {counter}")

            # Prepare feature vector for model prediction
            input_features = np.array([angle]).reshape(1, -1)
            predicted_feedback_encoded = model.predict(input_features)[0]
            predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]

            # Display the feedback
            cv2.putText(image, f"Feedback: {predicted_feedback}", 
                        (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

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
        cv2.putText(image, stage if stage else 'None', 
                    (60, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Mediapipe Feed', image)

        # Exit or pause functionality
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause
            print("Paused. Press 'p' again to resume.")
            while cv2.waitKey(10) & 0xFF != ord('p'):
                pass

cap.release()
cv2.destroyAllWindows()

#---------------------------------------newcode with log function
# import os
# import joblib
# import cv2
# import mediapipe as mp
# import numpy as np
# from scripts.utils import save_exercise_log  # Import the utility function
# import sys

# # Default user ID if not provided
# user_id = sys.argv[1] if len(sys.argv) > 1 else 'default_user'  # Set default if no user ID is passed

# # Define the angle calculation function
# def calculate_angle(a, b, c):
#     a = np.array(a)  # First point
#     b = np.array(b)  # Midpoint
#     c = np.array(c)  # End point

#     # Calculate vectors
#     ba = a - b
#     bc = c - b

#     # Calculate cosine similarity and angle
#     cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical issues
#     angle = np.arccos(cos_angle) * (180.0 / np.pi)  # Convert radians to degrees

#     return angle

# # Function to track bicep curl exercise
# def track_bicep_curl(user_id):
#     exercise_name = "Bicep Curl"  # Set exercise name for logging
#     counter = 0
#     stage = None

#     # Initialize MediaPipe Pose and Drawing modules
#     mp_pose = mp.solutions.pose
#     mp_drawing = mp.solutions.drawing_utils

#     # Initialize the video capture (using webcam)
#     cap = cv2.VideoCapture(0)

#     # Get screen resolution (optional)
#     screen_width = 1920  # Example for 1920px width
#     screen_height = 1080  # Example for 1080px height

#     # Load the trained model and label encoder
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(base_dir, 'mlp_feedback_model.pkl')
#     label_encoder_path = os.path.join(base_dir, 'feedback_label_encoder.pkl')

#     if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#         model = joblib.load(model_path)
#         label_encoder = joblib.load(label_encoder_path)
#     else:
#         raise FileNotFoundError("One or both .pkl files are missing in the 'scripts/bicep' directory.")

#     # Start Pose tracking with MediaPipe
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         # Create a named window and resize it to screen resolution
#         cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('Mediapipe Feed', screen_width, screen_height)

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Flip the frame horizontally for a mirror effect
#             frame = cv2.flip(frame, 1)

#             # Convert the image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False

#             # Process the image with MediaPipe Pose
#             results = pose.process(image)

#             # Convert back to BGR for rendering
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             try:
#                 # Extract landmarks
#                 landmarks = results.pose_landmarks.landmark

#                 # Normalize landmarks for angle calculation
#                 def get_coords(landmark):
#                     return [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]

#                 left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
#                 left_elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
#                 left_wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

#                 # Calculate the angle of the left elbow
#                 angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

#                 # Display the angle
#                 cv2.putText(image, f"Angle: {int(angle)}", 
#                             (50, 50), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Curl counting logic
#                 if angle > 160:  # Arm fully extended
#                     stage = "down"
#                 if angle < 30 and stage == "down":  # Arm fully curled
#                     stage = "up"
#                     counter += 1
#                     print(f"Reps: {counter}")

#                     # Log the exercise data
#                     save_exercise_log(exercise_name, counter, user_id)

#                 # Prepare feature vector for model prediction
#                 input_features = np.array([angle]).reshape(1, -1)
#                 predicted_feedback_encoded = model.predict(input_features)[0]
#                 predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]

#                 # Display the feedback
#                 cv2.putText(image, f"Feedback: {predicted_feedback}", 
#                             (30, 100), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

#             except Exception as e:
#                 print("Error:", e)

#             # Render counter and stage
#             cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

#             # Display rep count and stage
#             cv2.putText(image, 'REPS', (15, 12), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, str(counter), 
#                         (10, 60), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#             cv2.putText(image, 'STAGE', (65, 12), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, stage if stage else 'None', 
#                         (60, 60), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#             # Display the frame
#             cv2.imshow('Mediapipe Feed', image)

#             # Exit or pause functionality
#             key = cv2.waitKey(10) & 0xFF
#             if key == ord('q'):  # Quit
#                 break
#             elif key == ord('p'):  # Pause
#                 print("Paused. Press 'p' again to resume.")
#                 while cv2.waitKey(10) & 0xFF != ord('p'):
#                     pass

#     cap.release()
#     cv2.destroyAllWindows()

# # Track bicep curl for the user ID passed as an argument
# track_bicep_curl(user_id)
