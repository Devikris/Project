import requests
from datetime import datetime

def save_exercise_log(exercise_name, repetition_count, user_id, base_url='http://127.0.0.1:8000/logs/'):
    """
    Sends exercise data to the backend API.
    
    Args:
        exercise_name (str): The name of the exercise.
        repetition_count (int): The number of repetitions completed.
        user_id (int): The ID of the user.
        base_url (str): The API endpoint URL. Defaults to 'http://127.0.0.1:8000/logs/'.
    """
    data = {
        "exercise_name": exercise_name,
        "repetition_count": repetition_count,
        "user": user_id,
        "date": datetime.now().strftime('%Y-%m-%d')  # Current date in 'YYYY-MM-DD' format
    }
    try:
        response = requests.post(base_url, json=data)
        if response.status_code == 201:
            print(f"Exercise log for {exercise_name} saved successfully!")
        else:
            print(f"Failed to save exercise log: {response.json()}")
    except Exception as e:
        print(f"Error sending exercise log: {e}")
