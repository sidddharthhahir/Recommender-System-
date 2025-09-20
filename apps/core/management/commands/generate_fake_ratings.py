# from django.core.management.base import BaseCommand
# from django.contrib.auth import get_user_model
# from apps.movies.models import Movie, Rating
# from apps.accounts.models import Profile
# import random
# from datetime import date, timedelta

# User = get_user_model()

# class Command(BaseCommand):
#     help = "Generate fake users and ratings for testing LightFM"

#     def add_arguments(self, parser):
#         parser.add_argument('--users', type=int, default=20, help='Number of fake users to create')
#         parser.add_argument('--per-user', type=int, default=40, help='Number of ratings per user')

#     def handle(self, *args, **options):
#         num_users = options['users']
#         num_ratings = options['per_user']
        
#         movies = list(Movie.objects.all())
#         if not movies:
#             self.stdout.write(self.style.ERROR("❌ No movies in database. Run 'python manage.py sync_tmdb_movies' first."))
#             return
        
#         genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        
#         self.stdout.write(f"🎬 Creating {num_users} fake users with {num_ratings} ratings each...")
        
#         for i in range(num_users):
#             user, created = User.objects.get_or_create(
#                 username=f"testuser{i+1}",
#                 defaults={
#                     "email": f"testuser{i+1}@example.com",
#                     "first_name": "Test",
#                     "last_name": f"User{i+1}"
#                 }
#             )
            
#             # Create a profile if it doesn’t already exist
#             if created or not hasattr(user, "profile"):
#                 Profile.objects.create(
#                     user=user,
#                     birth_date=date.today() - timedelta(days=random.randint(18*365, 60*365)),
#                     favorite_genres=random.sample(genres, k=random.randint(2, 4)),
#                     preferred_languages=['en', 'hi'],
#                     openness_to_new=random.randint(1, 10)
#                 )
            
#             # Generate ratings
#             chosen_movies = random.sample(movies, min(num_ratings, len(movies)))
#             ratings_to_create = []
            
#             for movie in chosen_movies:
#                 rating_value = random.randint(1, 5)

#                 # Bias ratings if user likes the movie’s genre
#                 if hasattr(user, 'profile') and user.profile.favorite_genres:
#                     movie_genres = list(movie.genres.values_list('name', flat=True))
#                     if any(g in user.profile.favorite_genres for g in movie_genres):
#                         rating_value = min(5, rating_value + 1)  # boost rating slightly
                
#                 ratings_to_create.append(
#                     Rating(user=user, movie=movie, rating=rating_value)
#                 )
            
#             # Bulk insert ratings
#             Rating.objects.bulk_create(ratings_to_create, ignore_conflicts=True)
            
#             self.stdout.write(
#                 self.style.SUCCESS(f"✅ User {user.username}: {len(ratings_to_create)} ratings added")
#             )
        
#         total_users = User.objects.count()
#         total_ratings = Rating.objects.count()
#         self.stdout.write(
#             self.style.SUCCESS(
#                 f"🎉 Finished! {total_users} users, {total_ratings} total ratings in DB"
#             )
#         )
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from apps.movies.models import Movie, Rating
from apps.accounts.models import Profile
import random


class Command(BaseCommand):
    help = "Generate fake users and ratings for testing"

    def add_arguments(self, parser):
        parser.add_argument("--users", type=int, default=10, help="Number of test users to create")
        parser.add_argument("--per-user", type=int, default=10, help="Ratings per user")

    def handle(self, *args, **options):
        num_users = options["users"]
        per_user = options["per_user"]

        movies = list(Movie.objects.all())
        if not movies:
            self.stdout.write(self.style.ERROR("No movies found in database. Run sync_movies first."))
            return

        self.stdout.write(f"🎬 Creating {num_users} fake users with {per_user} ratings each...")

        for i in range(1, num_users + 1):
            username = f"testuser{i}"
            user, created = User.objects.get_or_create(
                username=username,
                defaults={"email": f"{username}@example.com"}
            )
            if created:
                user.set_password("password123")
                user.save()

            # ✅ Ensure profile exists but DON'T create duplicates
            Profile.objects.get_or_create(user=user)

            # Add ratings
            chosen_movies = random.sample(movies, min(per_user, len(movies)))
            for movie in chosen_movies:
                Rating.objects.update_or_create(
                    user=user,
                    movie=movie,
                    defaults={"rating": random.randint(1, 5)}
                )

            self.stdout.write(self.style.SUCCESS(f"✅ {username}: {per_user} ratings added"))

        self.stdout.write(self.style.SUCCESS("🎉 Fake ratings generation complete"))