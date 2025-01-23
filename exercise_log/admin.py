from django.contrib import admin
from .models import ExerciseLog

@admin.register(ExerciseLog)
class ExerciseLogAdmin(admin.ModelAdmin):
    list_display = ('exercise_name', 'repetitions', 'user', 'date')  # Corrected field name
