from django.db import models
from django.contrib.auth import get_user_model
import json

User = get_user_model()

class Genre(models.Model):
    """Movie genres from TMDB"""
    tmdb_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=100)
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']

class Movie(models.Model):
    """Movie model with TMDB integration"""
    
    # TMDB fields
    tmdb_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=500)
    original_title = models.CharField(max_length=500, blank=True)
    overview = models.TextField(blank=True)
    release_date = models.DateField(null=True, blank=True)
    runtime = models.IntegerField(null=True, blank=True)
    
    # Ratings and popularity
    tmdb_rating = models.FloatField(default=0.0)
    tmdb_vote_count = models.IntegerField(default=0)
    popularity = models.FloatField(default=0.0)
    
    # Media
    poster_path = models.CharField(max_length=200, blank=True)
    backdrop_path = models.CharField(max_length=200, blank=True)
    
    # Metadata
    genres = models.ManyToManyField(Genre, blank=True)
    original_language = models.CharField(max_length=10, blank=True)
    adult = models.BooleanField(default=False)
    
    # Additional metadata (JSON fields for flexibility)
    cast = models.JSONField(default=list, blank=True)  # [{"name": "Actor", "character": "Role"}]
    crew = models.JSONField(default=list, blank=True)  # [{"name": "Director", "job": "Director"}]
    keywords = models.JSONField(default=list, blank=True)  # ["action", "thriller"]
    production_companies = models.JSONField(default=list, blank=True)
    
    # Internal fields
    is_active = models.BooleanField(default=True)
    last_updated = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Recommendation fields
    avg_user_rating = models.FloatField(default=0.0)
    total_user_ratings = models.IntegerField(default=0)
    recommendation_score = models.FloatField(default=0.0)  # Cached LightFM score
    
    def __str__(self):
        year = self.release_date.year if self.release_date else "Unknown"
        return f"{self.title} ({year})"
    
    @property
    def poster_url(self):
        if self.poster_path:
            return f"https://image.tmdb.org/t/p/w500{self.poster_path}"
        return None
    
    @property
    def backdrop_url(self):
        if self.backdrop_path:
            return f"https://image.tmdb.org/t/p/w1280{self.backdrop_path}"
        return None
    
    def get_directors(self):
        """Extract directors from crew data"""
        return [person['name'] for person in self.crew if person.get('job') == 'Director']
    
    def get_main_cast(self, limit=5):
        """Get main cast members"""
        return self.cast[:limit] if self.cast else []
    
    def get_genre_names(self):
        """Get list of genre names"""
        return list(self.genres.values_list('name', flat=True))
    
    class Meta:
        ordering = ['-popularity', '-tmdb_rating']
        indexes = [
            models.Index(fields=['tmdb_id']),
            models.Index(fields=['release_date']),
            models.Index(fields=['tmdb_rating']),
            models.Index(fields=['popularity']),
        ]

class Rating(models.Model):
    """User ratings for movies"""
    
    RATING_CHOICES = [
        (1, '1 - Terrible'),
        (2, '2 - Bad'),
        (3, '3 - Okay'),
        (4, '4 - Good'),
        (5, '5 - Excellent'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ratings')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='ratings')
    rating = models.IntegerField(choices=RATING_CHOICES)
    
    # Additional feedback
    review = models.TextField(blank=True)
    would_recommend = models.BooleanField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} rated {self.movie.title}: {self.rating}/5"
    
    class Meta:
        unique_together = ('user', 'movie')
        ordering = ['-created_at']

class Watchlist(models.Model):
    """User watchlists"""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watchlist')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='watchlisted_by')
    added_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.movie.title}"
    
    class Meta:
        unique_together = ('user', 'movie')
        ordering = ['-added_at']

class MovieInteraction(models.Model):
    """Track user interactions with movies"""
    
    INTERACTION_TYPES = [
        ('view', 'Viewed Details'),
        ('click', 'Clicked Poster'),
        ('search', 'Found via Search'),
        ('recommendation', 'From Recommendation'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='interactions')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='interactions')
    interaction_type = models.CharField(max_length=20, choices=INTERACTION_TYPES)
    
    # Context
    source_page = models.CharField(max_length=100, blank=True)  # home, search, recommendations
    session_id = models.CharField(max_length=100, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} {self.interaction_type} {self.movie.title}"
    
    class Meta:
        ordering = ['-created_at']