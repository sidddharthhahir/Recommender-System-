from django.urls import path
from movies import views as movie_views

urlpatterns = [
    path('', movie_views.home, name='home'),   # <--- named 'home'
    # other routes...
]