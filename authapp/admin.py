from django.contrib import admin
from authapp.models import (
    Contact,
    MembershipPlan,
    Enrollment,
    Trainer,
    Gallery,
    Attendance, 
    Biceptricep,
    Legs,
    Chests,
    Body,
    Shoulder,
    BMICalculator,
     # Add this if `HeartRate` is part of authapp
)

# Register your models here.
admin.site.register(Contact)
admin.site.register(MembershipPlan)
admin.site.register(Enrollment)
admin.site.register(Trainer)
admin.site.register(Gallery)
admin.site.register(Attendance)
admin.site.register(Biceptricep)
admin.site.register(Legs)
admin.site.register(Chests)
admin.site.register(Body)
admin.site.register(Shoulder)
admin.site.register(BMICalculator)
# Register HeartRate if it belongs to authapp
