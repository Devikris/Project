from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from authapp.models import Contact,MembershipPlan,Trainer,Enrollment,Gallery,Attendance,Biceptricep,BicepCurl,ShoulderPress,TricepDip,TricepPushdown,Legs,Squat,Deadlift,Chests,Body,Pushup,Lateralraise,Russiantwist,Lateralpulldown,Legraise,Shoulder,Benchpress,Hammercurl,Pullup,Inclinebenchpress,Declinebenchpress,Chestflymachine,Romaniandeadlift,Plank,Tbarrow,Hipthrust,Legextension
import os
import subprocess
from django.http import HttpResponse
import asyncio
from bleak import BleakClient

from .models import HealthData
from django.contrib.auth.decorators import login_required
# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from exercise_log.models import ExerciseLog

def Home(request):
    return render(request,"index.html")

def gallery(request):
    posts=Gallery.objects.all()
    context={"posts":posts}
    return render(request,"gallery.html",context)


def attendance(request):
    if not request.user.is_authenticated:
        messages.warning(request,"Please Login and Try Again")
        return redirect('/login')
    SelectTrainer=Trainer.objects.all()
    context={"SelectTrainer":SelectTrainer}
    if request.method=="POST":
        phonenumber=request.POST.get('PhoneNumber')
        Login=request.POST.get('logintime')
        Logout=request.POST.get('loginout')
        SelectWorkout=request.POST.get('workout')
        TrainedBy=request.POST.get('trainer')
        query=Attendance(phonenumber=phonenumber,Login=Login,Logout=Logout,SelectWorkout=SelectWorkout,TrainedBy=TrainedBy)
        query.save()
        messages.warning(request,"Attendace Applied Success")
        return redirect('/attendance')
    return render(request,"attendance.html",context)

@login_required
def profile(request):
    # Ensure the user is authenticated
    if not request.user.is_authenticated:
        messages.warning(request, "Please Login and Try Again")
        return redirect('/login')


def profile_view(request):
    # Fetch user data and attendance (already present)
    posts = User.objects.all()  # Replace with your actual user query
    attendance = Attendance.objects.all()  # Replace with your actual attendance query
    
    # Fetch exercise logs
    exercise_logs = ExerciseLog.objects.all()

    # Pass data to the template
    context = {
        'posts': posts,
        'attendance': attendance,
        'exercise_logs': exercise_logs,
    }
    return render(request, 'profile.html', context)


def signup(request):
    if request.method=="POST":
        username=request.POST.get('usernumber')
        email=request.POST.get('email')
        pass1=request.POST.get('pass1')
        pass2=request.POST.get('pass2')
      
        if len(username)>10 or len(username)<10:
            messages.info(request,"Phone Number Must be 10 Digits")
            return redirect('/signup')

        if pass1!=pass2:
            messages.info(request,"Password is not Matching")
            return redirect('/signup')
       
        try:
            if User.objects.get(username=username):
                messages.warning(request,"Phone Number is Taken")
                return redirect('/signup')
           
        except Exception as identifier:
            pass
        
        
        try:
            if User.objects.get(email=email):
                messages.warning(request,"Email is Taken")
                return redirect('/signup')
           
        except Exception as identifier:
            pass
        
        
        
        myuser=User.objects.create_user(username,email,pass1)
        myuser.save()
        messages.success(request,"User is Created Please Login")
        return redirect('/login')
        
        
    return render(request,"signup.html")




def handlelogin(request):
    if request.method=="POST":        
        username=request.POST.get('usernumber')
        pass1=request.POST.get('pass1')
        myuser=authenticate(username=username,password=pass1)
        if myuser is not None:
            login(request,myuser)
            messages.success(request,"Login Successful")
            return redirect('/')
        else:
            messages.error(request,"Invalid Credentials")
            return redirect('/login')
            
        
    return render(request,"handlelogin.html")


def handleLogout(request):
    logout(request)
    messages.success(request,"Logout Success")    
    return redirect('/login')

def contact(request):
    if request.method=="POST":
        name=request.POST.get('fullname')
        email=request.POST.get('email')
        number=request.POST.get('num')
        desc=request.POST.get('desc')
        myquery=Contact(name=name,email=email,phonenumber=number,description=desc)
        myquery.save()       
        messages.info(request,"Thanks for Contacting us we will get back you soon")
        return redirect('/contact')
        
    return render(request,"contact.html")


def enroll(request):
    if not request.user.is_authenticated:
        messages.warning(request,"Please Login and Try Again")
        return redirect('/login')

    Membership=MembershipPlan.objects.all()
    SelectTrainer=Trainer.objects.all()
    context={"Membership":Membership,"SelectTrainer":SelectTrainer}
    if request.method=="POST":
        FullName=request.POST.get('FullName')
        email=request.POST.get('email')
        gender=request.POST.get('gender')
        PhoneNumber=request.POST.get('PhoneNumber')
        DOB=request.POST.get('DOB')
        member=request.POST.get('member')
        trainer=request.POST.get('trainer')
        reference=request.POST.get('reference')
        address=request.POST.get('address')
        query=Enrollment(FullName=FullName,Email=email,Gender=gender,PhoneNumber=PhoneNumber,DOB=DOB,SelectMembershipplan=member,SelectTrainer=trainer,Reference=reference,Address=address)
        query.save()
        messages.success(request,"Thanks For Enrollment")
        return redirect('/join')



    return render(request,"enroll.html",context)

def biceps_pose(request):
    # Construct the absolute path to the biceps.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/bicep/bicepp.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
def hammercurl(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/hammercurl/hammercurl.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
def hipthrust(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/hipthrust/hipthrust.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def tbarrow(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/tbarrow/tbarrow.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def benchpress(request):
    # Construct the absolute path to the biceps.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/benchpress/benchpress.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
    
def chest(request):
    # Construct the absolute path to the chestt.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/chest/chestt.py')
    video_path = os.path.join(os.path.dirname(script_path), 'push-up_0.mp4')

    try:
        # Execute the script and capture the output
        result = subprocess.run(["python", script_path, video_path], check=True, capture_output=True, text=True)
        print(result.stdout)  # Log output for debugging
        return redirect('Home')
    except subprocess.CalledProcessError as e:
        # Log detailed error information
        return HttpResponse(f"Script execution failed. Return code: {e.returncode}, Output: {e.output}, Error: {e.stderr}")
    except Exception as e:
        # Handle generic errors
        return HttpResponse(f"Unexpected error occurred: {str(e)}")


def lateralraise(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/lateral_raise/lateralraise.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")   
def squats(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/squats/squats.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    

def legraise(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/legraise/legraise.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def legextension(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/legextension/legextension.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def shoulderpress(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/shoulderpress/shoulderpress.py')

    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def declinebenchpress(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/declinebenchpress/declinebenchpress.py')
    video_path = os.path.join(os.path.dirname(script_path), 'decline3.mp4')
    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def inclinebenchpress(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/inclinebenchpress/inclinebenchpress.py')
    video_path = os.path.join(os.path.dirname(script_path), 'decline3.mp4')
    try:
        # Use subprocess to execute the script
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def pullup(request):
    # Construct the absolute path to the lateralraise.py script
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/pullup/pullup.py')

    video_path = os.path.join(os.path.dirname(script_path), 'pullup2.mp4')  # Video path in the same folder

    try:
        # Use subprocess to execute the script with the video path as an argument
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    

WATCH_MAC_ADDRESS = "31:E5:D2:5E:3B:19"  # Replace with your MAC address


# Fetch the heart rate from the smartwatch and save it to the user's profile
async def fetch_and_save_heart_rate(user):
    async with BleakClient(WATCH_MAC_ADDRESS) as client:
        if client.is_connected:
            print("Connected to Smartwatch")
            
            heart_rate_uuid = "00002a19-0000-1000-8000-00805f9b34fb"
            
            # Read the heart rate data
            heart_rate_data = await client.read_gatt_char(heart_rate_uuid)
            
            # Decode and convert the heart rate data (assuming it's in bytes)
            heart_rate = int.from_bytes(heart_rate_data, byteorder="little")
            
            # Save the heart rate data to the database
            health_data = HealthData(user=user, heart_rate=heart_rate)
            health_data.save()
            
            print(f"Heart rate saved: {heart_rate} bpm")
            return heart_rate
        else:
            raise ConnectionError("Failed to connect to the smartwatch.")


# Profile view that shows enrollment details, attendance, and heart rate
@login_required
def profile_view(request):
    if request.method == "POST":
        # Fetch and save heart rate data when the user triggers it
        try:
            heart_rate = asyncio.run(fetch_and_save_heart_rate(request.user))
            return render(request, "profile.html", {"heart_rate": heart_rate})
        except Exception as e:
            print(f"Error fetching heart rate: {e}")
            return render(request, "profile.html", {"error": "Failed to fetch heart rate."})
    else:
        # Display user's profile with saved heart rate data
        health_data = HealthData.objects.filter(user=request.user).order_by('-date_time')
        return render(request, "profile.html", {
            "attendance": request.user.attendance_set.all(),  # Assuming you have related attendance data
            "posts": request.user.enrollment_set.all(),  # Assuming you have related enrollment data
            "health_data": health_data
        })
    

# def russiantwist(request):
#     # Construct the absolute path to the lateralraise.py script
#     script_path = os.path.join(os.path.dirname(__file__), '../scripts/russian_twist/russiantwist.py')

#     try:
#         # Use subprocess to execute the script
#         subprocess.run(["python", script_path], check=True)
        
#         # Redirect to the homepage or any other view you want
#         return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
#     except Exception as e:
#         # Handle errors gracefully and return the error message
#         return HttpResponse(f"Error occurred: {str(e)}")

def russiantwist(request):
    # Construct the absolute path to the script and the video file
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/russian_twist/russiantwist.py')
    video_path = os.path.join(os.path.dirname(script_path), 'russian10.mp4')  # Video path in the same folder

    try:
        # Use subprocess to execute the script with the video path as an argument
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    

def tricepdips(request):
    # Construct the absolute path to the script and the video file
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/tricepdips/tricepdips.py')
    video_path = os.path.join(os.path.dirname(script_path), 'tricepdips_8.mp4')  # Video path in the same folder

    try:
        # Use subprocess to execute the script with the video path as an argument
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")


def triceppushdown(request):
    # Construct the absolute path to the script and the video file
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/triceppushdown/triceppushdown.py')
    video_path = os.path.join(os.path.dirname(script_path), 'triceppushdown_1.mp4')  # Video path in the same folder

    try:
        # Use subprocess to execute the script with the video path as an argument
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")



def deadlift(request):
    # Construct the absolute path to the script and the video file
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/romaniandeadlift/romaniandeadlift.py')
    video_path = os.path.join(os.path.dirname(script_path), 'romaniandeadlift_5.mp4')  # Video path in the same folder

    try:
        # Use subprocess to execute the script with the video path as an argument
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")

# from django.shortcuts import render
# from .models import ExerciseLog
def romaniandeadlift(request):
    # Construct the absolute path to the script and the video file
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/romaniandeadlift/romaniandeadlift.py')
    video_path = os.path.join(os.path.dirname(script_path), 'romaniandeadlift_5.mp4')  # Video path in the same folder

    try:
        # Use subprocess to execute the script with the video path as an argument
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
def profile_view(request):
    # Fetch exercise logs for the logged-in user
    exercise_logs = ExerciseLog.objects.filter(user=request.user).order_by('-date')  # Most recent logs first
    return render(request, 'profile.html', {'exercise_logs': exercise_logs})

def lateralpulldown(request):
    # Construct the absolute path to the script and the video file
    script_path = os.path.join(os.path.dirname(__file__), '../scripts/lateralpulldown/lateralpulldown.py')
    video_path = os.path.join(os.path.dirname(script_path), 'latpulldown1.mp4')  # Video path in the same folder

    try:
        # Use subprocess to execute the script with the video path as an argument
        subprocess.run(["python", script_path, video_path], check=True)
        
        # Redirect to the homepage or any other view you want
        return redirect('Home')  # 'Home' is the name of the homepage view or URL pattern
    except Exception as e:
        # Handle errors gracefully and return the error message
        return HttpResponse(f"Error occurred: {str(e)}")
    
def exercise_list(request):
    biceptricep=Biceptricep.objects.all()
    return render(request,'biceptricep.html',{'biceptricep':biceptricep})
def exercises_legs(request):
    legs=Legs.objects.all()
    return render(request,'legs.html',{'legs':legs})
def exercises_chests(request):
    chests=Chests.objects.all()
    return render(request,'chests.html',{'chests':chests})
def exercises_body(request):
    body=Body.objects.all()
    return render(request,'body.html',{'body':body})
def exercises_shoulder(request):
    body=Shoulder.objects.all()
    return render(request,'shoulder.html',{'shoulder':Shoulder})
def tobicepcurlpage(request):
    biceptricep=BicepCurl.objects.all()
    return render(request,'curlbicep.html',{'tobicepcurlpage':tobicepcurlpage})
def toshoulderpresspage(request):
    shoulder=ShoulderPress.objects.all()
    return render(request,'shoulderpress1.html',{'toshoulderpresspage':toshoulderpresspage})
def totricepdippage(request):
    biceptricep=TricepDip.objects.all()
    return render(request,'tricepdip1.html',{'totricepdippage':totricepdippage})
def totriceppushdownpage(request):
    biceptricep=TricepDip.objects.all()
    return render(request,'triceppushdown1.html',{'totriceppushdownpage':totriceppushdownpage})
def tosquatpage(request):
    legs=Squat.objects.all()
    return render(request, 'squat1.html',{'tosquatpage':tosquatpage})
def todeadliftpage(request):
    legs=Deadlift.objects.all()
    return render(request, 'deadlift1.html',{'todeadliftpage':todeadliftpage})
def tolegraisepage(request):
    legs=Legraise.objects.all()
    return render(request, 'legraise1.html',{'tolegraisepage':tolegraisepage})
def torussiantwistpage(request):
    body=Russiantwist.objects.all()
    return render(request, 'russiantwist1.html',{'torussiantwistpage':torussiantwistpage})
def tolateralpulldownpage(request):
    body=Lateralpulldown.objects.all()
    return render(request, 'lateralpulldown1.html',{'tolateralpulldownpage':tolateralpulldownpage})
def topushuppage(request):
    chests=Pushup.objects.all()
    return render(request, 'pushup1.html',{'topushuppage':topushuppage})
def tolateralraisepage(request):
    shoulder=Lateralraise.objects.all()
    return render(request, 'lateralraise1.html',{'tolateralraisepage':tolateralraisepage})
def tobenchpresspage(request):
    chests=Benchpress.objects.all()
    return render(request, 'benchpress1.html',{'tobenchpresspage':tobenchpresspage})
def tohammercurlpage(request):
    biceptricep=Hammercurl.objects.all()
    return render(request, 'hammercurl1.html',{'tohammercurlpage':tohammercurlpage})
def topulluppage(request):
    biceptricep=Pullup.objects.all()
    return render(request, 'pullup1.html',{'topulluppage':topulluppage})

def toinclinebenchpress(request):
    shoulder=Inclinebenchpress.objects.all()
    return render(request, 'inclinebenchpress1.html',{'toinclinebenchpress':toinclinebenchpress})

def todeclinebenchpress(request):
    shoulder=Declinebenchpress.objects.all()
    return render(request, 'declinebenchpress1.html',{'todeclinebenchpress':todeclinebenchpress})

def tochestflymachine(request):
    shoulder=Chestflymachine.objects.all()
    return render(request, 'chestflymachine1.html',{'tochestflymachine':tochestflymachine})

def toromaniandeadlift(request):
    shoulder=Romaniandeadlift.objects.all()
    return render(request, 'romaniandeadlift1.html',{'toromaniandeadlift':toromaniandeadlift})
def toplank(request):
    shoulder=Plank.objects.all()
    return render(request, 'plank.html',{'toplank':toplank})

def totbarrow(request):
    shoulder=Tbarrow.objects.all()
    return render(request, 'tbarrow1.html',{'totbarrow':totbarrow})

def tohipthrust(request):
    shoulder=Hipthrust.objects.all()
    return render(request, 'hipthrust1.html',{'tohipthrust':tohipthrust})

def tolegextension(request):
    shoulder=Legextension.objects.all()
    return render(request, 'legextension1.html',{'tolegextension':tolegextension})


