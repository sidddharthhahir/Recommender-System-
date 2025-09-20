# from django.db import models
# from django.contrib.auth.models import User


# class Profile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")

#     # Fields for personalization & explanations
#     birth_date = models.DateField(null=True, blank=True)   # instead of `date_of_birth`
#     favorite_genres = models.JSONField(default=list, blank=True)  # e.g. ["Action", "Comedy"]
#     preferred_languages = models.JSONField(default=list, blank=True)  # e.g. ["en", "hi"]
#     openness_to_new = models.IntegerField(null=True, blank=True)  # Scale 1–10
    
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     @property
#     def survey_completed(self):
#         return bool(self.favorite_genres and self.preferred_languages and self.openness_to_new)

#     def __str__(self):
#         return f"Profile of {self.user.username}"

#     @property
#     def age(self):
#         """Calculate age from birth_date (for explainability & grouping)"""
#         from datetime import date
#         if self.birth_date:
#             return (date.today() - self.birth_date).days // 365
#         return None

#     @property
#     def age_group(self):
#         """Return age group label"""
#         if not self.age:
#             return None
#         age = self.age
#         if age < 25:
#             return "young"
#         elif age < 35:
#             return "adult"
#         elif age < 50:
#             return "middle"
#         return "senior"


#     def get_preference_vector(self):
#         """Return user preferences as a feature vector"""
#         return {
#             "favorite_genres": self.favorite_genres,
#             "favorite_actors": self.favorite_actors,
#             "openness_to_new": self.openness_to_new,
#             "min_rating_threshold": self.min_rating_threshold,
#         }


# class UserPreference(models.Model):
#     """Detailed user survey preferences"""

#     profile = models.OneToOneField(
#     Profile,
#     on_delete=models.CASCADE,
#     related_name="preferences",
#     null=True,
#     blank=True
# )

#     # Survey responses
#     favorite_movie = models.CharField(max_length=200, blank=True)
#     least_favorite_genre = models.JSONField(default=list)
#     movie_frequency = models.CharField(max_length=50, blank=True)  # daily, weekly, monthly
#     discovery_method = models.CharField(max_length=100, blank=True)  # friends, critics, algorithms

#     # Implicit preferences (learned)
#     genre_weights = models.JSONField(default=dict)
#     actor_weights = models.JSONField(default=dict)
#     director_weights = models.JSONField(default=dict)

#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     def __str__(self):
#         return f"{self.profile.user.username}'s preferences"
from django.db import models
from django.contrib.auth.models import User
from datetime import date


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")

    # Core personalization fields
    birth_date = models.DateField(null=True, blank=True)
    favorite_genres = models.JSONField(default=list, blank=True)  # e.g. ["Action", "Comedy"]
    preferred_languages = models.JSONField(default=list, blank=True)  # e.g. ["en", "hi"]
    openness_to_new = models.IntegerField(null=True, blank=True)  # Scale 1–10

    # Extended survey fields
    favorite_actor = models.CharField(max_length=255, blank=True, null=True)
    last_loved_movie = models.CharField(max_length=255, blank=True, null=True)

    RATING_CHOICES = [
        ("generous", "Generous"),
        ("balanced", "Balanced"),
        ("critical", "Critical"),
    ]
    rating_style = models.CharField(
        max_length=20,
        choices=RATING_CHOICES,
        default="balanced",
        blank=True
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def survey_completed(self):
        return bool(self.favorite_genres and self.preferred_languages and self.openness_to_new)

    def __str__(self):
        return f"Profile of {self.user.username}"

    @property
    def age(self):
        if self.birth_date:
            return (date.today() - self.birth_date).days // 365
        return None

    @property
    def age_group(self):
        if not self.age:
            return None
        if self.age < 25:
            return "young"
        elif self.age < 35:
            return "adult"
        elif self.age < 50:
            return "middle"
        return "senior"

    def get_preference_vector(self):
        """Return user survey preferences in a clean dictionary"""
        return {
            "favorite_genres": self.favorite_genres,
            "favorite_actor": self.favorite_actor,
            "last_loved_movie": self.last_loved_movie,
            "rating_style": self.rating_style,
            "openness_to_new": self.openness_to_new,
            "preferred_languages": self.preferred_languages,
        }


class UserPreference(models.Model):
    """Extra user preference experiments (optional)"""

    profile = models.OneToOneField(
        Profile,
        on_delete=models.CASCADE,
        related_name="preferences",
        null=True,
        blank=True
    )

    # Survey responses (extended version)
    favorite_movie = models.CharField(max_length=200, blank=True)
    least_favorite_genre = models.JSONField(default=list, blank=True)
    movie_frequency = models.CharField(max_length=50, blank=True)  # daily, weekly, monthly
    discovery_method = models.CharField(max_length=100, blank=True)  # friends, critics, algorithms

    # Implicit (learned from behavior)
    genre_weights = models.JSONField(default=dict, blank=True)
    actor_weights = models.JSONField(default=dict, blank=True)
    director_weights = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        if self.profile:
            return f"{self.profile.user.username}'s preferences"
        return "Orphan UserPreference"