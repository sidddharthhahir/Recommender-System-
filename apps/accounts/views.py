from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as django_logout
from django.http import JsonResponse
import json
from django.urls import reverse
from apps.recommender.models import Movie
from apps.movies.models import Movie


from .models import Profile

GENRES_CHOICES = [
    "Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller",
    "Adventure", "Fantasy", "Animation", "Documentary", "Crime", "History", "Family"
]

LANG_CHOICES = ["en", "hi", "fr", "es"]


# --- SIGNUP ---
def home(request):
    # movies = Movie.objects.all()[:20]
    movies = Movie.objects.all().order_by("-popularity")
    # return render(request, "home.html")
    return render(request, "home.html", {"movies": movies})
def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            Profile.objects.get_or_create(user=user)
            login(request, user)
            return redirect("accounts:post_login_redirect")
    else:
        form = UserCreationForm()
    return render(request, "accounts/signup.html", {"form": form})


# --- REGISTER (alias) ---
def register(request):
    return signup(request)


# --- LOGIN ---
def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("accounts:post_login_redirect")
    else:
        form = AuthenticationForm()
    return render(request, "accounts/login.html", {"form": form})


# --- LOGOUT ---
@login_required
def logout_view(request):
    django_logout(request)
    return redirect("accounts:login")


# --- SURVEY (page) ---
@login_required
def survey(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)
    return render(request, "accounts/survey.html", {
        "profile": profile,
        "genres": GENRES_CHOICES,
        "languages": LANG_CHOICES,
    })


# --- API for survey (AJAX from AlpineJS) ---
# @login_required
# def survey_api(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body.decode("utf-8"))
#             profile, _ = Profile.objects.get_or_create(user=request.user)

#             # Save data from JSON
#             profile.favorite_genres = data.get("favorite_genres", [])
#             profile.preferred_languages = data.get("preferred_languages", ["en"])
#             profile.openness_to_new = int(data.get("openness_to_new", 5))

#             # Optional fields from extended survey
#             if "favorite_actor" in data:
#                 profile.favorite_actor = data.get("favorite_actor")
#             if "last_loved_movie" in data:
#                 profile.last_loved_movie = data.get("last_loved_movie")
#             if "rating_style" in data:
#                 profile.rating_style = data.get("rating_style")

#             profile.save()

#             return JsonResponse({
#     "success": True,
#     "message": "Survey saved ✅",
#     "redirect_url": "/recommender/recommendations/"
# })
#         except Exception as e:
#             return JsonResponse({"success": False, "message": str(e)}, status=400)

#     return JsonResponse({"success": False, "message": "Invalid request"}, status=405)
@login_required
def survey_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            profile, _ = Profile.objects.get_or_create(user=request.user)

            profile.favorite_genres = data.get("favorite_genres", [])
            profile.preferred_languages = data.get("preferred_languages", ["en"])
            profile.openness_to_new = int(data.get("openness_to_new", 5))

            if "favorite_actor" in data:
                profile.favorite_actor = data.get("favorite_actor")
            if "last_loved_movie" in data:
                profile.last_loved_movie = data.get("last_loved_movie")
            if "rating_style" in data:
                profile.rating_style = data.get("rating_style")

            profile.save()

            # Use reverse to generate the URL (replace 'home' with your named home URL)
            return JsonResponse({
                "success": True,
                "message": "Survey saved ✅",
                "redirect_url": reverse("home")
            })
        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)}, status=400)

    return JsonResponse({"success": False, "message": "Invalid request"}, status=405)



# --- PROFILE (edit later) ---
@login_required
def profile_view(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)

    if request.method == "POST":
        selected_genres = request.POST.getlist("genres")
        selected_langs = request.POST.getlist("languages")
        openness = request.POST.get("openness")

        profile.favorite_genres = selected_genres
        profile.preferred_languages = selected_langs
        profile.openness_to_new = int(openness) if openness and openness.isdigit() else None
        profile.save()

        return redirect("accounts:profile")

    return render(request, "accounts/profile.html", {
        "profile": profile,
        "genres": GENRES_CHOICES,
        "languages": LANG_CHOICES,
    })


# --- POST LOGIN REDIRECT ---
# @login_required
# def post_login_redirect(request):
#     profile, _ = Profile.objects.get_or_create(user=request.user)
#     if not profile.favorite_genres or not profile.preferred_languages or not profile.openness_to_new:
#         return redirect("accounts:survey")
#     # return redirect("recommender:recommendations")
#     return redirect("home")
@login_required
def post_login_redirect(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)

    # If survey not filled
    if not profile.favorite_genres or not profile.preferred_languages or not profile.openness_to_new:
        return redirect("accounts:survey")

    # Otherwise always go to home
    return redirect("home")