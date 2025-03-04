import os
import joblib
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import matplotlib.pyplot as plt
import sys  # Import sys to get command-line arguments

# Check if a video path is provided

# Text-to-speech engine setup
global counter, stage
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define paths to the model and label encoder
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'mlp_model_inclined_benchpress.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder_inclined_benchpress.pkl')

# Load trained model and label encoder
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
else:
    raise FileNotFoundError("One or both .pkl files are missing for Incline Bench Press.")
# Video Source Selection
if len(sys.argv) > 1:
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)  # Load video
else:
    cap = cv2.VideoCapture(0)  # Default to webcam

if not cap.isOpened():
    print("Error: Unable to open video file or webcam.")
    sys.exit() # Default to webcam if no video provided

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
elapsed_time=0

# cap = cv2.VideoCapture(0)
counter, stage = 0, None

last_motivation_time = 0
cool_off_seconds = 5

cv2.namedWindow('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
for i in range(cool_off_seconds, 0, -1):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(frame, f"Starting in {i}s...", (700, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.imshow('Mediapipe Feed', frame)
    if i==5:
           speak(f"Starting in {i}")
    else:
           speak(f"{i}")
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
speak("Start!")
timer_start = time.time()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    feedback_counts = {}
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # frame = cv2.flip(frame, 1)
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

# Extract necessary landmarks
                    # Initialize variables if not already set
                    # Initialize variables globally if not already set
                    # Ensure counter and stage are globally defined
                    

# Initialize if not already set
                  

# Extract key points
                    left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                    left_elbow = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
                    left_wrist = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

# Calculate elbow angle
                    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

# Debugging: Print tracking details
                    print(f"Before Checking: Elbow Angle: {elbow_angle}, Stage: {stage}, Count: {counter}")

# Ensure initial stage assignment
                    if stage is None:
                        stage = "down"

# **Fix the Transition Logic**
                    if elbow_angle > 150:  # Fully extended
                        stage="down"
                    if elbow_angle < 90 and stage=="down":
                        stage="up"
                        counter+=1
                        speak(f"{counter}")
                    print(f"Repetition Count Updated: {counter}")

# Debugging - Track changes after execution
                    




                    input_features = np.array([elbow_angle]).reshape(1, -1)
                    predicted_feedback_encoded = model.predict(input_features)[0]
                    predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]
                    print(f"Feedback: {predicted_feedback}")

                    if predicted_feedback in feedback_counts:
                        feedback_counts[predicted_feedback] += 1
                    else:
                        feedback_counts[predicted_feedback] = 1

                    if "Great form! You're reaching the top range!" in predicted_feedback:
                        correct_count += 1
                    elif "Perfect incline bench press form!" in predicted_feedback:
                        correct_count += 1
                    elif "" in predicted_feedback: 
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
        
            
# Left Rectangle (Repetitions Counter)
                    cv2.rectangle(image, (0, 0), (200, 73), (50, 50, 50), -1)  # Dark gray rectangle on left
                    cv2.putText(image, 'REPETITIONS', (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Moved slightly up
                    cv2.putText(image, str(counter), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)  # Increased gap

# Right Rectangle (Timer)
                    right_x_start = image.shape[1] - 200
                    cv2.rectangle(image, (right_x_start, 0), (image.shape[1], 73), (50, 50, 50), -1)  # Dark gray rectangle on right

# Timer Heading
                    cv2.putText(image, 'TIMER', (right_x_start + 50, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Centered text

# Timer Value
                    minutes, seconds = divmod(elapsed_time, 60)
                    timer_text = f"{minutes:02}:{seconds:02}"
                    cv2.putText(image, timer_text, (right_x_start + 50, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Increased size & gap

                    cv2.imshow('Mediapipe Feed', image)
            except Exception as e:
                print("Error:", e)

            # cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            # cv2.putText(image, 'REPETITIONS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            # minutes, seconds = divmod(elapsed_time, 60)
            # timer_text = f"{minutes:02}:{seconds:02}"
            # cv2.putText(image, timer_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

            # cv2.imshow('Mediapipe Feed', image)

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
            cv2.imshow('Mediapipe Feed', frame)
            cv2.putText(frame, f"Feedback: {most_frequent_feedback}", (500, 600), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Mediapipe Feed', frame)
            speak(f"While performing the session in future, consider to {most_frequent_feedback}")
            speak("Press Q to move to see the  Performance graph")

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
            paused = not paused  # Toggle pause state
            if paused:
                pause_start_time = time.time()
                speak("Session paused. Press P to resume.")
            else:
                speak(f"Session resumed! You are at {minutes} minutes {seconds} seconds.")
                timer_start += time.time() - pause_start_time  # Adjust timer to maintain elapsed time
                

# Add a pause loop inside the while loop
        while paused:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('p'):
               speak(f"Session resumed! You are at {minutes} minutes {seconds} seconds.")
               paused = False
               timer_start += time.time() - pause_start_time
               
               break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cap.release()
    cv2.destroyAllWindows()