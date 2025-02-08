#tricep dips with opencv
# import os
# import joblib
# import cv2
# import mediapipe as mp
# import numpy as np

# # Define the path to the 'scripts/tricep_dips' directory
# base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
# model_path = os.path.join(base_dir, 'mlp_model_tricepdips.pkl')
# label_encoder_path = os.path.join(base_dir, 'label_encoder_tricepdips.pkl')

# # Load the trained model and label encoder
# if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#     model = joblib.load(model_path)
#     label_encoder = joblib.load(label_encoder_path)
# else:
#     raise FileNotFoundError("One or both .pkl files are missing in the 'scripts/tricep_dips' directory.")

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

# # Counter and stage variables for tricep dips
# counter = 0
# stage = None
# paused = False  # Pause functionality flag

# # Set up MediaPipe Pose instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     # Create a named window and resize it to screen resolution
#     cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Mediapipe Feed', screen_width, screen_height)

#     while cap.isOpened():
#         if not paused:
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
#                     return [landmark.x, landmark.y]

#                 # Points for tricep dips
#                 left_elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
#                 left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
#                 left_wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

#                 # Calculate the elbow angle
#                 elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

#                 # Display the angle
#                 cv2.putText(image, f"Elbow Angle: {int(elbow_angle)}",
#                             tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Tricep dip counting logic based on elbow angle
#                 if elbow_angle > 160:
#                     stage = "up"
#                 if elbow_angle < 90 and stage == "up":
#                     stage = "down"
#                     counter += 1
#                     print(f"Reps: {counter}")

#                 # Prepare feature vector for model prediction
#                 input_features = np.array([elbow_angle]).reshape(1, -1)
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
#             cv2.putText(image, stage if stage else "None",
#                         (65, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#             # Render detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # Show the frame
#         cv2.imshow('Mediapipe Feed', image)

#         # Handle keyboard input for quitting or pausing
#         key = cv2.waitKey(10) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('p'):
#             paused = not paused

#     cap.release()
#     cv2.destroyAllWindows()




# import os
# import joblib
# import cv2
# import mediapipe as mp
# import numpy as np
# import sys

# # Check if video path is passed as an argument
# if len(sys.argv) < 2:
#     raise ValueError("Please provide the path to the video file as an argument.")

# video_path = sys.argv[1]
# print(f"Video path: {video_path}")

# # Define the path to the model and label encoder
# base_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(base_dir, 'mlp_model_tricepdips.pkl')
# label_encoder_path = os.path.join(base_dir, 'label_encoder_tricepdips.pkl')

# # Load the trained model and label encoder
# if os.path.exists(model_path) and os.path.exists(label_encoder_path):
#     model = joblib.load(model_path)
#     label_encoder = joblib.load(label_encoder_path)
# else:
#     raise FileNotFoundError("One or both .pkl files are missing.")

# # Define the angle calculation function
# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     # Calculate vectors
#     ba = a - b
#     bc = c - b

#     # Calculate cosine similarity and angle
#     cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     cos_angle = np.clip(cos_angle, -1.0, 1.0)
#     angle = np.arccos(cos_angle) * (180.0 / np.pi)  # Convert radians to degrees
#     return angle

# # Initialize MediaPipe Pose and Drawing modules
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# # Open the video file
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     raise FileNotFoundError(f"Unable to open video file: {video_path}")

# # Counter and stage variables
# counter = 0
# stage = None
# paused = False

# # Set up MediaPipe Pose instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         if not paused:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Flip the frame horizontally
#             frame = cv2.flip(frame, 1)

#             # Convert the image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False

#             # Process the image with MediaPipe Pose
#             results = pose.process(image)

#             # Convert back to BGR
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             try:
#                 # Ensure landmarks are detected
#                 if not results.pose_landmarks:
#                     raise ValueError("Landmarks not detected.")

#                 # Extract landmarks
#                 landmarks = results.pose_landmarks.landmark

#                 # Normalize landmarks for angle calculation
#                 def get_coords(landmark):
#                     return [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]

#                 # Points for tricep dips posture
#                 shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
#                 elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
#                 wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

#                 # Calculate the elbow angle
#                 elbow_angle = calculate_angle(shoulder, elbow, wrist)
#                 print(f"Elbow Angle: {elbow_angle}")  # Debugging

#                 # Display the angle
#                 cv2.putText(image, f"Elbow Angle: {int(elbow_angle)}",
#                             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Tricep dips counting logic
#                 if elbow_angle > 160 and stage != "up":  # "up" position
#                     stage = "up"
#                     print(f"Stage set to 'up'. Angle: {elbow_angle}")  # Debugging

#                 if elbow_angle < 90 and stage == "up":  # "down" position
#                     counter += 1  # Increment counter when dip is complete
#                     stage = "down"
#                     print(f"Repetition counted: {counter}. Stage set to 'down'. Angle: {elbow_angle}")  # Debugging

#                 # Prepare feature vector for model prediction
#                 input_features = np.array([elbow_angle]).reshape(1, -1)
#                 predicted_feedback_encoded = model.predict(input_features)[0]
#                 predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]

#                 # Display the feedback
#                 cv2.putText(image, f"Feedback: {predicted_feedback}",
#                             (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 117, 16), 2, cv2.LINE_AA)

#             except Exception as e:
#                 print("Error:", e)

#             # Render counter and stage
#             cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

#             # Display rep count and stage
#             cv2.putText(image, 'REPS', (15, 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, str(counter),
#                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
#             cv2.putText(image, 'STAGE', (65, 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#             cv2.putText(image, stage if stage else "None",
#                         (65, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

#             # Render detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # Show the frame
#         cv2.imshow('Mediapipe Feed', image)

#         key = cv2.waitKey(10) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('p'):
#             paused = not paused  # Toggle pause

#     cap.release()
#     cv2.destroyAllWindows()
import os
import joblib
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import matplotlib.pyplot as plt

# Text-to-speech engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define paths to the model and label encoder
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'mlp_model_tricepdips.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder_tricepdips.pkl')

# Load trained model and label encoder
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
else:
    raise FileNotFoundError("One or both .pkl files are missing for Triceps Dips.")
screen_width = 1920  # Example for 1920px width
screen_height = 1080  # Example for 1080px height

def plot_graph(correct, incorrect):
    labels = ['Correct', 'Incorrect']
    counts = [correct, incorrect]
    colors = ['#4CAF50', '#FF5733']  # Green for correct, red for incorrect

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color=colors)
    plt.title('Performance Metrics')
    plt.xlabel('Feedback')
    plt.ylabel('Count')
    plt.show()

# Initialize performance metrics
correct_count = 0
incorrect_count = 0
elapsed_time = 0
paused = False
pause_start_time = 0

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180.0 / np.pi)
    return angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
counter, stage = 0, None
timer_start = time.time()
last_motivation_time = 0
cool_off_seconds = 5

cv2.namedWindow('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
for i in range(cool_off_seconds, 0, -1):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(frame, f"Starting in {i}s...", (700, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.imshow('Mediapipe Feed', frame)
    speak(f"{i}")
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
speak("Start!")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    feedback_counts = {}
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    def get_coords(landmark):
                        return [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]

                    left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                    left_elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                    left_wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

                    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    cv2.putText(image, f"Elbow Angle: {int(elbow_angle)}",
                                tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    if elbow_angle > 160:
                        stage = "up"
                    if elbow_angle < 90 and stage == "up":
                        stage = "down"
                        counter += 1
                        speak(f"{counter}")
                        print(f"Reps: {counter}")

                    input_features = np.array([elbow_angle]).reshape(1, -1)
                    predicted_feedback_encoded = model.predict(input_features)[0]
                    predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]
                    print(f"Feedback: {predicted_feedback}")

                    if predicted_feedback in feedback_counts:
                        feedback_counts[predicted_feedback] += 1
                    else:
                        feedback_counts[predicted_feedback] = 1

                    if "Good" in predicted_feedback:
                       correct_count += 1
                    else:
                        incorrect_count += 1

                    elapsed_time = int(time.time() - timer_start)
                    if elapsed_time // 30 > last_motivation_time // 30:
                        last_motivation_time = elapsed_time
                        minutes, seconds = divmod(elapsed_time, 60)
                        speak(f"{minutes} minute {seconds} seconds completed!")

                    feedback_x, feedback_y = frame.shape[1] // 2 - 300, frame.shape[0] - 50
                    cv2.putText(image, predicted_feedback, (feedback_x, feedback_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print("Error:", e)

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPETITIONS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            minutes, seconds = divmod(elapsed_time, 60)
            timer_text = f"{minutes:02}:{seconds:02}"
            cv2.putText(image, timer_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

            cv2.imshow('Mediapipe Feed', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            most_frequent_feedback = max(feedback_counts, key=feedback_counts.get)
            speak("Session Over Here are your performance report.")

            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(frame, "Performance Report", (600, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            cv2.imshow('Mediapipe Feed', frame)
            speak("Performance Report")
            cv2.waitKey(1000)

            cv2.putText(frame, f"Total Time: {minutes} min {seconds} sec", (500, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imshow('Mediapipe Feed', frame)
            speak(f"Total time taken is {minutes} minutes and {seconds} seconds")
            cv2.waitKey(1000)

            cv2.putText(frame, f"Repetitions: {counter}", (500, 500), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imshow('Mediapipe Feed', frame)
            speak(f"The total repetitions are {counter} times")
            speak("Press Q to move to see the  Performance graph")
            cv2.imshow('Mediapipe Feed', frame)
            cv2.putText(frame, f"Feedback: {most_frequent_feedback}", (500, 600), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Mediapipe Feed', frame)
            speak(f"While performing the session in future, consider to {most_frequent_feedback}")
            

            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    plot_graph(correct_count, incorrect_count)
                    speak("Press Q to exit.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # cv2.putText(frame, f"Feedback: {random_feedback}", (500, 600), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # cv2.imshow('Mediapipe Feed', frame)
            # speak(f"While performing the session in future, consider to {random_feedback}")
            # cv2.putText(frame, f"Feedback: {most_frequent_feedback}", (500, 600), 
            # cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # cv2.imshow('Mediapipe Feed', frame)
            # speak(f"While performing the session in future, consider to {most_frequent_feedback}")
            
        elif key == ord('p'):
            if paused:
                paused = False
                timer_start += time.time() - pause_start_time
                speak(f"Session resumed! You are at {minutes} minutes {seconds} seconds.")
            else:
                paused = True
                pause_start_time = time.time()
                speak("Session paused. Press P to resume.")

    cap.release()
    cv2.destroyAllWindows()
      