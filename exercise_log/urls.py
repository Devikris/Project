from django.urls import path
from .views import ExerciseLogView

urlpatterns = [
    path('', ExerciseLogView.as_view(), name='exercise-log'),  # Base route for logs
]
