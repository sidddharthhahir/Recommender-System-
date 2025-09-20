from django.shortcuts import render
# from movies.models import Movie
from apps.movies.models import Movie
from .models import Movie

def home(request):
    movies = Movie.objects.all().order_by('title')
    return render(request, 'home.html', {'movies': movies})