from rest_framework import serializers
from .models import ExerciseLog

class ExerciseLogSerializer(serializers.ModelSerializer):
    date = serializers.DateTimeField(format='%Y-%m-%dT%H:%M:%S')  # Explicitly define the format
    
    class Meta:
        model = ExerciseLog
        fields = ['id', 'user', 'exercise_name', 'repetition_count', 'date']
