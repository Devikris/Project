# exercise_log/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import ExerciseLog
from .serializers import ExerciseLogSerializer

# exercise_log/views.py
class ExerciseLogView(APIView):
    def get(self, request):
        # Filter logs by the currently authenticated user
        logs = ExerciseLog.objects.filter(user=request.user)
        serializer = ExerciseLogSerializer(logs, many=True)
        return Response(serializer.data)

    def post(self, request):
        # Automatically associate the log with the currently authenticated user
        data = request.data.copy()
        data['user'] = request.user.id  # Add the user ID to the data
        serializer = ExerciseLogSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
