from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

def home(request):
    if not request.user.is_authenticated:
        return render(request, "home.html")  # guest landing page

    profile = request.user.profile
    if not profile.survey_completed:   # ✅ using the property in Profile
        return redirect("accounts:survey")

    return redirect("recommender:recommendations")