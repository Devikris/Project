import os
import joblib
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import matplotlib.pyplot as plt
# import random

# Text-to-speech engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume level

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define the path to the 'scripts/hammercurl' directory
base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
model_path = os.path.join(base_dir, 'mlp_model_squats.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder_squats.pkl')

# Load the trained model and label encoder
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
else:
    raise FileNotFoundError("One or both .pkl files are missing in the 'scripts/hammercurl' directory.")

# Get screen resolution (optional)
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

# Main program loop
is_active = False
correct_count = 0
incorrect_count = 0
elapsed_time=0

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
# Counter and stage variables for hammer curl
counter = 0
stage = None
timer_start = None
last_motivation_time = 0
cool_off_seconds = 5
  # Cool-off period

# Cool-off countdown
cv2.namedWindow('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
for i in range(cool_off_seconds, 0, -1):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Black screen
    cv2.putText(frame, f"Starting in {i}s...", (700, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.imshow('Mediapipe Feed', frame)
    if i == 5:
        speak(f"Starting in {i} ")
    else:
        speak(f"{i}")
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
speak("Start!")

# Counter and stage variables for hammer curl
counter = 0
stage = None
timer_start = time.time()  # Start the timer when the program starts
last_motivation_time = 0  # To track the last motivational message time

# Random feedback for improvement
# feedback_list = [
#     "Try to keep your back straight during the curls!",
#     "Focus on controlling your movement to improve form!",
#     "Keep your elbows steady and close to your torso!",
#     "Avoid swinging your arms for better results!",
#     "Maintain a consistent pace for effective repetitions!"
# ]

# Set up MediaPipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cv2.namedWindow('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    elapsed_time_at_pause = 0
    # Dictionary to track feedback counts
    feedback_counts = {}

    while cap.isOpened():
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
            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark

            def get_coords(landmark):
                return [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]

           
            left_knee = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
            left_hip = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
            left_ankle = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

            # Calculate the knee angle
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # Display the angle
            cv2.putText(image, f"Knee Angle: {int(knee_angle)}", 
                        tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Squat counting logic based on knee angle
            if knee_angle > 160:
                stage = "up"
            if knee_angle < 90 and stage == "up":
                stage = "down"
                counter += 1
                print(f"Reps: {counter}")

            input_features = np.array([angle]).reshape(1, -1)
            predicted_feedback_encoded = model.predict(input_features)[0]
            predicted_feedback = label_encoder.inverse_transform([predicted_feedback_encoded])[0]
            print(f"Feedback: {predicted_feedback}")
            

            # Update feedback counts
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
                minutes = elapsed_time // 60
                seconds = elapsed_time % 60
                speak(f"{minutes} minute {seconds} seconds completed!")

            feedback_x = frame.shape[1] // 2 - 300
            feedback_y = frame.shape[0] - 50
            cv2.putText(image, predicted_feedback, 
                        (feedback_x, feedback_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)

        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        cv2.putText(image, 'REPETITIONS', (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        timer_radius = 50
        center = (frame.shape[1] - 80, 80)
        cv2.circle(image, center, timer_radius, (0, 0, 0), -1)

        angle = (elapsed_time % 60) * 6
        cv2.ellipse(image, center, (timer_radius, timer_radius), 90, 0, angle, (255, 255, 255), -1)

        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        timer_text = f"{minutes:02}:{seconds:02}"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(image, timer_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)

        cv2.imshow('Mediapipe Feed', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            total_time = int(time.time() - timer_start)
            minutes = total_time // 60
            seconds = total_time % 60

            # random_feedback = random.choice(feedback_list)
            # Find the most frequent feedback
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
            cv2.waitKey(1000)

            # cv2.putText(frame, f"Feedback: {random_feedback}", (500, 600), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # cv2.imshow('Mediapipe Feed', frame)
            # speak(f"While performing the session in future, consider to {random_feedback}")
            cv2.putText(frame, f"Feedback: {most_frequent_feedback}", (500, 600), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Mediapipe Feed', frame)
            speak(f"While performing the session in future, consider to {most_frequent_feedback}")
            cv2.waitKey(1000)

            cv2.putText(frame, "Press Q to see Performance Graph.", (500, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)#(length,height)
            speak("Press Q to move to see the  Performance graph")
            cv2.imshow('Mediapipe Feed', frame)

            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    plot_graph(correct_count, incorrect_count)
                    speak("Press Q to exit.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        elif key == ord('p'):
            speak("Hold on! Session paused. Press P to get back on track.")
            elapsed_time_at_pause = int(time.time() - timer_start)
            while cv2.waitKey(10) & 0xFF != ord('p'):
                pass
            minutes = elapsed_time_at_pause // 60
            seconds = elapsed_time_at_pause % 60
            speak(f"Session Resumed You're back in action after {minutes} minutes and {seconds} seconds of workout")
    
    cap.release()
    cv2.destroyAllWindows()
