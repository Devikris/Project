# from distutils.command.upload import upload
# from django.db import models

# Create your models here.
# this file we creates a database tables

# class Contact(models.Model):
#     name=models.CharField(max_length=25)
#     email=models.EmailField()
#     phonenumber=models.CharField(max_length=12)
#     description=models.TextField()

#     def __str__(self):
#         return self.email

# class Enrollment(models.Model):        
#     FullName=models.CharField(max_length=25)
#     Email=models.EmailField()
#     Gender=models.CharField(max_length=25)
#     PhoneNumber=models.CharField(max_length=12)
#     DOB=models.CharField(max_length=50)
#     SelectMembershipplan=models.CharField(max_length=200)
#     SelectTrainer=models.CharField(max_length=55)
#     Reference=models.CharField(max_length=55)
#     Address=models.TextField()
#     paymentStatus=models.CharField(max_length=55,blank=True,null=True)
#     Price=models.IntegerField(max_length=55,blank=True,null=True)
#     DueDate=models.DateTimeField(blank=True,null=True)
#     timeStamp=models.DateTimeField(auto_now_add=True,blank=True,)

#     def __str__(self):
#         return self.FullName

# class Trainer(models.Model):
#     name=models.CharField(max_length=55)
#     gender=models.CharField(max_length=25)
#     phone=models.CharField(max_length=25)
#     salary=models.IntegerField(max_length=25)
#     timeStamp=models.DateTimeField(auto_now_add=True,blank=True)
#     def __str__(self):
#         return self.name

# class MembershipPlan(models.Model):
#     plan=models.CharField(max_length=185)
#     price=models.IntegerField(max_length=55)

#     def __int__(self):
#         return self.id


# class Gallery(models.Model):
#     title=models.CharField(max_length=100)
#     img=models.ImageField(upload_to='gallery')
#     timeStamp=models.DateTimeField(auto_now_add=True,blank=True)
#     def __int__(self):
#         return self.id


# class Attendance(models.Model):
#     Selectdate=models.DateTimeField(auto_now_add=True)
#     phonenumber=models.CharField(max_length=15)
#     Login=models.CharField(max_length=200)
#     Logout=models.CharField(max_length=200)
#     SelectWorkout=models.CharField(max_length=200)
#     TrainedBy=models.CharField(max_length=200)
#     def __int__(self):
#         return self.id
from django.db import models
from django.contrib.auth.models import User

class HealthData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    heart_rate = models.IntegerField()  # Stores heart rate data
    date_time = models.DateTimeField(auto_now_add=True)  # Records the time when the data was fetched
    
    def __str__(self):
        return f"{self.user.username} - {self.heart_rate} bpm"


# Create your models here.
# this file creates database tables

# 





# Example of other models (unchanged for brevity)
class Contact(models.Model):
    name = models.CharField(max_length=25)
    email = models.EmailField()
    phonenumber = models.CharField(max_length=12)
    description = models.TextField()

    def __str__(self):
        return self.email

class Enrollment(models.Model):        
    FullName = models.CharField(max_length=25)
    Email = models.EmailField()
    Gender = models.CharField(max_length=25)
    PhoneNumber = models.CharField(max_length=12)
    DOB = models.CharField(max_length=50)
    SelectMembershipplan = models.CharField(max_length=200)
    SelectTrainer = models.CharField(max_length=55)
    Reference = models.CharField(max_length=55)
    Address = models.TextField()
    paymentStatus = models.CharField(max_length=55, blank=True, null=True)
    Price = models.IntegerField(blank=True, null=True)
    DueDate = models.DateTimeField(blank=True, null=True)
    timeStamp = models.DateTimeField(auto_now_add=True, blank=True)

    def __str__(self):
        return self.FullName
class Trainer(models.Model):
    name = models.CharField(max_length=55)
    gender = models.CharField(max_length=25)
    phone = models.CharField(max_length=25)
    salary = models.IntegerField()
    timeStamp = models.DateTimeField(auto_now_add=True, blank=True)

    def __str__(self):
        return self.name

class MembershipPlan(models.Model):
    plan = models.CharField(max_length=185)
    price = models.IntegerField()

    def __int__(self):
        return self.id

class Gallery(models.Model):
    title = models.CharField(max_length=100)
    img = models.ImageField(upload_to='gallery')  # Correct Django handling of uploads
    timeStamp = models.DateTimeField(auto_now_add=True, blank=True)

    def __int__(self):
        return self.id

class Attendance(models.Model):
    Selectdate = models.DateTimeField(auto_now_add=True)
    phonenumber = models.CharField(max_length=15)
    Login = models.CharField(max_length=200)
    Logout = models.CharField(max_length=200)
    SelectWorkout = models.CharField(max_length=200)
    TrainedBy = models.CharField(max_length=200)

    def __int__(self):
        return self.id


    from django.db import models
from django.contrib.auth.models import User

class ExerciseLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='authapp_exercise_logs')
    exercise_name = models.CharField(max_length=100)
    repetitions = models.IntegerField()
    date = models.DateField(auto_now_add=True)

class Biceptricep(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Legs(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Chests(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Body(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Shoulder(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class BicepCurl(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class ShoulderPress(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class TricepDip(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class TricepPushdown(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Squat(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Deadlift(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Pushup(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Lateralraise(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Russiantwist(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Lateralpulldown(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
class Legraise(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)    
class Benchpress(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50) 
class Hammercurl(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)

class Pullup(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)


class Inclinebenchpress(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)

class Declinebenchpress(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)

class Chestflymachine(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)

class Romaniandeadlift2(models.Model):
    name=models.CharField(max_length=100)
    category=models.CharField(max_length=50)
    