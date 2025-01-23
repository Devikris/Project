from django.db import models
from django.contrib.auth.models import User

class ExerciseLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='exercise_log_exercise_logs')
    exercise_name = models.CharField(max_length=100)
    repetitions = models.IntegerField()
    date = models.DateField(auto_now_add=True)
