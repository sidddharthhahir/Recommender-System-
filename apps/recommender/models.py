from django.db import models
from django.contrib.auth import get_user_model
from apps.movies.models import Movie, Rating
from django.contrib.auth.models import User
import json

User = get_user_model()

class RecommendationLog(models.Model):
    """Log of recommendations served to users"""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recommendation_logs')
    movies = models.ManyToManyField(Movie, through='RecommendationItem')
    
    # Request context
    request_type = models.CharField(max_length=50, default='homepage')  # homepage, search, similar
    algorithm_version = models.CharField(max_length=20, default='lightfm_v1')
    
    # Metadata
    total_movies = models.IntegerField(default=0)
    avg_confidence = models.FloatField(default=0.0)
    processing_time_ms = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Recommendations for {self.user.username} at {self.created_at}"
    
    class Meta:
        ordering = ['-created_at']

class RecommendationItem(models.Model):
    """Individual recommendation within a log"""
    
    log = models.ForeignKey(RecommendationLog, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    
    # Recommendation details
    position = models.IntegerField()  # 1, 2, 3... (ranking position)
    confidence_score = models.FloatField()  # LightFM prediction score
    
    # User interaction
    clicked = models.BooleanField(default=False)
    clicked_at = models.DateTimeField(null=True, blank=True)
    rated = models.BooleanField(default=False)
    rated_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.movie.title} (pos {self.position}, score {self.confidence_score:.3f})"
    
    class Meta:
        ordering = ['position']
        unique_together = ('log', 'movie')

class ExplanationLog(models.Model):
    """Log of explanations shown to users"""
    
    EXPLANATION_TYPES = [
        ('shap', 'SHAP Global'),
        ('lime', 'LIME Local'),
        ('anchor', 'Anchor Rules'),
        ('counterfactual', 'Counterfactual'),
        ('combined', 'Combined View'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='explanation_logs')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='explanation_logs')
    recommendation_log = models.ForeignKey(RecommendationLog, on_delete=models.CASCADE, null=True, blank=True)
    
    # Explanation details
    explanation_type = models.CharField(max_length=20, choices=EXPLANATION_TYPES)
    explanation_data = models.JSONField()  # Store the actual explanation
    
    # User interaction
    viewed = models.BooleanField(default=True)  # True when explanation is shown
    clicked_details = models.BooleanField(default=False)  # True when user clicks for more details
    helpful_rating = models.IntegerField(null=True, blank=True)  # 1-5 scale
    
    # Metadata
    generation_time_ms = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.explanation_type} for {self.movie.title} to {self.user.username}"
    
    class Meta:
        ordering = ['-created_at']

class ModelVersion(models.Model):
    """Track different versions of the recommendation model"""
    
    version_name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=50, default="lightfm")
    metrics = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Training details
    training_data_size = models.IntegerField()
    num_users = models.IntegerField(default=0)
    num_items = models.IntegerField(default=0)
    num_interactions = models.IntegerField(default=0)
    num_epochs = models.IntegerField(default=20)
    num_components = models.IntegerField(default=50)
    learning_rate = models.FloatField(default=0.05)
    loss_function = models.CharField(max_length=50, default='warp')
    training_duration_minutes = models.FloatField(default=0.0)
    hyperparameters = models.JSONField(default=dict)
    
    # Performance metrics
    precision_at_10 = models.FloatField(null=True, blank=True)
    recall_at_10 = models.FloatField(null=True, blank=True)
    auc_score = models.FloatField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=False)
    model_file_path = models.CharField(max_length=500, blank=True)
    
    def __str__(self):
        status = "ACTIVE" if self.is_active else "INACTIVE"
        return f"{self.version_name} ({self.algorithm}) - {status}"
    
    class Meta:
        ordering = ['-created_at']

class UserEmbedding(models.Model):
    """Store user embeddings from LightFM"""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='embedding')
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    
    # Embedding vector (stored as JSON array)
    embedding_vector = models.JSONField()
    
    # Metadata
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Embedding for {self.user.username}"

class MovieEmbedding(models.Model):
    """Store movie embeddings from LightFM"""
    
    movie = models.OneToOneField(Movie, on_delete=models.CASCADE, related_name='embedding')
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE)
    
    # Embedding vector (stored as JSON array)
    embedding_vector = models.JSONField()
    
    # Metadata
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Embedding for {self.movie.title}"

class SurveyResponse(models.Model):
    """Store cold-start survey responses"""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='survey_response')
    
    # Survey questions and answers
    favorite_genres = models.JSONField()  # List of genre IDs
    favorite_actor = models.CharField(max_length=200, blank=True)
    last_loved_movie = models.CharField(max_length=200, blank=True)
    rating_style = models.CharField(max_length=50, blank=True)  # generous, critical, balanced
    openness_to_new = models.IntegerField()  # 1-10 scale
    
    # Generated seed ratings
    seed_ratings_generated = models.BooleanField(default=False)
    seed_ratings_count = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Survey response from {self.user.username}"

class Movie(models.Model):
    title = models.CharField(max_length=255)
    genres = models.CharField(max_length=255, blank=True, null=True)
    tmdb_id = models.IntegerField(unique=True)
    tmdb_rating = models.FloatField(blank=True, null=True)
    poster_url = models.URLField(blank=True, null=True)
    overview = models.TextField(blank=True, null=True)
    release_date = models.DateField(blank=True, null=True)
    runtime = models.IntegerField(blank=True, null=True)  # in minutes
    budget = models.BigIntegerField(blank=True, null=True)
    revenue = models.BigIntegerField(blank=True, null=True)
    vote_count = models.IntegerField(default=0)
    popularity = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-popularity', '-tmdb_rating']

    def __str__(self):
        return self.title

    @property
    def genre_list(self):
        """Return genres as a list"""
        if self.genres:
            return [genre.strip() for genre in self.genres.split('|')]
        return []

    @property
    def rating_display(self):
        """Display rating with one decimal place"""
        if self.tmdb_rating:
            return f"{self.tmdb_rating:.1f}"
        return "N/A"

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.FloatField()  # 1.0 to 5.0
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'movie')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.movie.title}: {self.rating}"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=30, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Simple preference system
    preferred_genres = models.TextField(blank=True, help_text="Comma-separated list of preferred genres")
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    @property
    def badges(self):
        """Return preferred genres as a list for compatibility"""
        if self.preferred_genres:
            return [{'name': genre.strip()} for genre in self.preferred_genres.split(',')]
        return []

class ExplanationHistory(models.Model):
    EXPLANATION_TYPES = (
        ('lightfm', 'LightFM'),
        ('baseline', 'Baseline'),
        ('collaborative', 'Collaborative Filtering'),
        ('content', 'Content-Based'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)  # Now Movie exists!
    explanation_type = models.CharField(max_length=50, choices=EXPLANATION_TYPES, default='lightfm')
    explanation_data = models.JSONField()
    generation_time_ms = models.IntegerField(default=0)
    helpful_rating = models.IntegerField(null=True, blank=True, choices=[(i, i) for i in range(1, 6)])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Explanation Histories"

    def __str__(self):
        return f"{self.user.username} - {self.movie.title} ({self.explanation_type})"

    def get_explanation_type_display_custom(self):
        return dict(self.EXPLANATION_TYPES).get(self.explanation_type, self.explanation_type)
    