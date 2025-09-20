from django.core.management.base import BaseCommand
from django.conf import settings
import requests
import logging
from apps.movies.models import Movie, Genre

logger = logging.getLogger('recommender')


class Command(BaseCommand):
    help = 'Sync movie data from TMDB API (popular movies, language-specific)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--seed',
            type=int,
            help='Number of popular movies to seed initially',
        )
        parser.add_argument(
            '--update',
            action='store_true',
            help='Update existing movies with latest data',
        )
        parser.add_argument(
            '--lang',
            type=str,
            default="en",
            help='Original language filter (e.g., en for English, hi for Hindi, te for Telugu)',
        )

    def handle(self, *args, **options):
        lang = options['lang']

        if options['seed']:
            self.seed_popular_movies(options['seed'], lang)
        elif options['update']:
            self.update_existing_movies()
        else:
            self.stdout.write(
                self.style.ERROR("Please specify --seed <number> or --update")
            )

    def seed_popular_movies(self, count, lang):
        """Seed database with popular movies from TMDB"""
        self.stdout.write(f"Seeding {count} popular movies (lang={lang}) from TMDB...")

        # First sync genres
        self.sync_genres()

        movies_synced = 0
        page = 1

        while movies_synced < count:
            try:
                # Use discover endpoint so we can filter by language
                response = requests.get(
                    f"{settings.TMDB_BASE_URL}/discover/movie",
                    params={
                        "api_key": settings.TMDB_API_KEY,
                        "page": page,
                        "language": "en-US",  # metadata in English
                        "with_original_language": lang,
                        "sort_by": "popularity.desc",
                    },
                )
                response.raise_for_status()
                data = response.json()

                for movie_data in data.get("results", []):
                    if movies_synced >= count:
                        break

                    movie_detail = self.get_movie_details(movie_data["id"])
                    if movie_detail:
                        self.create_or_update_movie(movie_detail)
                        movies_synced += 1

                        if movies_synced % 50 == 0:
                            self.stdout.write(f"Synced {movies_synced} movies...")

                page += 1

            except requests.RequestException as e:
                logger.error(f"TMDB API error: {e}")
                break

        self.stdout.write(
            self.style.SUCCESS(
                f"✅ Successfully synced {movies_synced} movies (lang={lang})"
            )
        )

    def sync_genres(self):
        """Sync movie genres from TMDB"""
        self.stdout.write("Syncing genres from TMDB...")

        try:
            response = requests.get(
                f"{settings.TMDB_BASE_URL}/genre/movie/list",
                params={"api_key": settings.TMDB_API_KEY, "language": "en-US"},
            )
            response.raise_for_status()
            data = response.json()

            for genre_data in data.get("genres", []):
                Genre.objects.get_or_create(
                    tmdb_id=genre_data["id"],
                    defaults={"name": genre_data["name"]},
                )

            self.stdout.write(
                self.style.SUCCESS(f"Synced {len(data['genres'])} genres")
            )

        except requests.RequestException as e:
            logger.error(f"Genre sync error: {e}")

    def get_movie_details(self, movie_id):
        """Get detailed movie information from TMDB"""
        try:
            response = requests.get(
                f"{settings.TMDB_BASE_URL}/movie/{movie_id}",
                params={
                    "api_key": settings.TMDB_API_KEY,
                    "language": "en-US",
                    "append_to_response": "credits,keywords",
                },
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Movie details error for ID {movie_id}: {e}")
            return None

    def create_or_update_movie(self, movie_data):
        """Create or update movie in database"""
        try:
            movie, created = Movie.objects.get_or_create(
                tmdb_id=movie_data["id"],
                defaults={
                    "title": movie_data.get("title", ""),
                    "original_title": movie_data.get("original_title", ""),
                    "overview": movie_data.get("overview", ""),
                    "release_date": movie_data.get("release_date") or None,
                    "runtime": movie_data.get("runtime"),
                    "tmdb_rating": movie_data.get("vote_average", 0.0),
                    "tmdb_vote_count": movie_data.get("vote_count", 0),
                    "popularity": movie_data.get("popularity", 0.0),
                    "poster_path": movie_data.get("poster_path") or "",
                    "backdrop_path": movie_data.get("backdrop_path") or "",
                    "original_language": movie_data.get("original_language", "en"),
                    "adult": movie_data.get("adult", False),
                },
            )

            if not created:
                # Update some fields
                movie.title = movie_data.get("title", movie.title)
                movie.overview = movie_data.get("overview", movie.overview)
                movie.tmdb_rating = movie_data.get("vote_average", movie.tmdb_rating)
                movie.tmdb_vote_count = movie_data.get("vote_count", movie.tmdb_vote_count)
                movie.popularity = movie_data.get("popularity", movie.popularity)
                movie.save()

            # Set genres
            if "genres" in movie_data:
                genre_ids = [g["id"] for g in movie_data["genres"]]
                genres = Genre.objects.filter(tmdb_id__in=genre_ids)
                movie.genres.set(genres)

            # Save cast
            if "credits" in movie_data:
                credits = movie_data["credits"]

                # Cast (top 10)
                cast_data = [
                    {"name": p["name"], "character": p.get("character", "")}
                    for p in credits.get("cast", [])[:10]
                ]
                movie.cast = cast_data

                # Crew
                crew_data = []
                for person in credits.get("crew", []):
                    if person.get("job") in ["Director", "Writer", "Producer", "Screenplay"]:
                        crew_data.append({"name": person["name"], "job": person["job"]})
                movie.crew = crew_data

            # Keywords
            if "keywords" in movie_data:
                keywords = [
                    kw["name"]
                    for kw in movie_data.get("keywords", {}).get("keywords", [])
                ]
                movie.keywords = keywords[:20]

            movie.save()

        except Exception as e:
            logger.error(
                f"Error creating/updating movie {movie_data.get('title', 'Unknown')}: {e}"
            )

    def update_existing_movies(self):
        """Update existing movies with latest TMDB data"""
        self.stdout.write("Updating existing movies...")

        movies = Movie.objects.filter(is_active=True)
        updated_count = 0

        for movie in movies:
            movie_detail = self.get_movie_details(movie.tmdb_id)
            if movie_detail:
                self.create_or_update_movie(movie_detail)
                updated_count += 1

                if updated_count % 20 == 0:
                    self.stdout.write(f"Updated {updated_count} movies...")

        self.stdout.write(
            self.style.SUCCESS(f"✅ Successfully updated {updated_count} movies")
        )