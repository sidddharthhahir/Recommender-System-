from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Avg
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import time

from .models import Movie, Rating, UserProfile, ExplanationHistory
from .engine import engine

logger = logging.getLogger(__name__)

@login_required
def recommendations(request):
    """Generate and display personalized movie recommendations"""
    try:
        # Load model if not already loaded
        if not engine.model:
            model_path = 'recommender/models/lightfm_model.pkl'
            if not engine.load_model(model_path):
                # Train new model if loading fails
                if not engine.train():
                    messages.error(request, "Could not generate recommendations. Please try again later.")
                    return render(request, 'recommender/recommendations.html', {'recommendations': []})
        
        # Get recommendations
        recommendations_data = engine.predict(request.user.id, num_recommendations=12)
        
        # Extract movies and add confidence scores
        recommendations = []
        seen_movie_ids = set()
        
        for rec_data in recommendations_data:
            movie = rec_data['movie']
            # Skip if we've already added this movie (deduplication)
            if movie.id in seen_movie_ids:
                continue
            seen_movie_ids.add(movie.id)
            
            # Add confidence score to movie object for template
            movie.confidence_score = rec_data['confidence_score']
            recommendations.append(movie)
        
        # If no personalized recommendations, get popular movies
        if not recommendations:
            recommendations = Movie.objects.annotate(
                avg_rating=Avg('rating__rating')
            ).filter(
                avg_rating__isnull=False
            ).order_by('-avg_rating')[:12]
            
            # Add default confidence scores
            for movie in recommendations:
                movie.confidence_score = 0.5
        
        context = {
            'recommendations': recommendations,
            'user': request.user
        }
        
        return render(request, 'recommender/recommendations.html', context)
        
    except Exception as e:
        logger.error(f"Error in recommendations view: {e}")
        messages.error(request, "An error occurred while generating recommendations.")
        return render(request, 'recommender/recommendations.html', {'recommendations': []})

@login_required
@require_http_methods(["GET", "POST"])
def explain_recommendation(request, movie_id):
    """Generate explanation for a movie recommendation"""
    try:
        movie = get_object_or_404(Movie, id=movie_id)
        
        if request.method == "POST" or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # AJAX request for inline explanation
            start_time = time.time()
            
            # Generate explanation
            explanation_data = engine.explain(request.user.id, movie_id)
            
            generation_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            
            # Save explanation to history
            try:
                ExplanationHistory.objects.create(
                    user=request.user,
                    movie=movie,
                    explanation_type='lightfm',
                    explanation_data=explanation_data,
                    generation_time_ms=generation_time
                )
            except Exception as e:
                logger.warning(f"Could not save explanation history: {e}")
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # Return JSON for AJAX
                return JsonResponse({
                    'success': True,
                    'explanation': explanation_data,
                    'movie_title': movie.title,
                    'generation_time': generation_time
                })
            else:
                # Regular POST request
                context = {
                    'movie': movie,
                    'explanation': explanation_data,
                    'generation_time': generation_time
                }
                return render(request, 'recommender/explain.html', context)
        
        else:
            # GET request - show explanation page
            context = {
                'movie': movie,
                'explanation': None
            }
            return render(request, 'recommender/explain.html', context)
            
    except Exception as e:
        logger.error(f"Error in explain_recommendation view: {e}")
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': False,
                'error': 'Could not generate explanation. Please try again.'
            })
        else:
            messages.error(request, "Could not generate explanation.")
            return redirect('recommender:recommendations')

@login_required
def movie_detail(request, movie_id):
    """Display detailed information about a movie"""
    movie = get_object_or_404(Movie, id=movie_id)
    
    # Get user's rating if exists
    user_rating = None
    try:
        user_rating = Rating.objects.get(user=request.user, movie=movie)
    except Rating.DoesNotExist:
        pass
    
    # Handle rating form submission
    if request.method == 'POST':
        rating_value = request.POST.get('rating')
        if rating_value:
            try:
                rating_value = float(rating_value)
                if 1 <= rating_value <= 5:
                    # Update or create rating
                    rating, created = Rating.objects.update_or_create(
                        user=request.user,
                        movie=movie,
                        defaults={'rating': rating_value}
                    )
                    
                    messages.success(request, f"Your rating of {rating_value} stars has been saved!")
                    return redirect('recommender:movie_detail', movie_id=movie_id)
                else:
                    messages.error(request, "Rating must be between 1 and 5.")
            except ValueError:
                messages.error(request, "Invalid rating value.")
    
    # Get average rating
    avg_rating = Rating.objects.filter(movie=movie).aggregate(Avg('rating'))['rating__avg']
    
    context = {
        'movie': movie,
        'user_rating': user_rating,
        'avg_rating': round(avg_rating, 1) if avg_rating else None,
        'total_ratings': Rating.objects.filter(movie=movie).count()
    }
    
    return render(request, 'recommender/movie_detail.html', context)

@login_required
def movie_list(request):
    """Display paginated list of all movies with search functionality"""
    query = request.GET.get('q', '')
    genre_filter = request.GET.get('genre', '')
    
    movies = Movie.objects.all()
    
    # Apply search filter
    if query:
        movies = movies.filter(
            Q(title__icontains=query) |
            Q(genres__icontains=query) |
            Q(overview__icontains=query)
        )
    
    # Apply genre filter
    if genre_filter:
        movies = movies.filter(genres__icontains=genre_filter)
    
    # Order by popularity/rating
    movies = movies.annotate(
        avg_rating=Avg('rating__rating')
    ).order_by('-avg_rating', '-tmdb_rating')
    
    # Pagination
    paginator = Paginator(movies, 24)  # Show 24 movies per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get unique genres for filter dropdown
    all_genres = set()
    for movie in Movie.objects.exclude(genres__isnull=True).exclude(genres=''):
        if movie.genres:
            genres = movie.genres.split('|')
            all_genres.update(genres)
    
    context = {
        'page_obj': page_obj,
        'query': query,
        'genre_filter': genre_filter,
        'all_genres': sorted(all_genres),
        'total_movies': movies.count()
    }
    
    return render(request, 'recommender/movie_list.html', context)

@login_required
def user_profile(request):
    """Display and manage user profile"""
    # Get or create user profile
    try:
        profile = request.user.userprofile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)
    
    # Handle profile updates
    if request.method == 'POST':
        bio = request.POST.get('bio', '')
        location = request.POST.get('location', '')
        preferred_genres = request.POST.get('preferred_genres', '')
        
        profile.bio = bio
        profile.location = location
        profile.preferred_genres = preferred_genres
        profile.save()
        
        messages.success(request, "Profile updated successfully!")
        return redirect('recommender:user_profile')
    
    # Get user's ratings
    user_ratings = Rating.objects.filter(user=request.user).select_related('movie').order_by('-created_at')[:10]
    
    # Get user's explanation history
    explanations = ExplanationHistory.objects.filter(user=request.user).select_related('movie').order_by('-created_at')[:5]
    
    context = {
        'profile': profile,
        'user_ratings': user_ratings,
        'recent_explanations': explanations,
        'total_ratings': Rating.objects.filter(user=request.user).count(),
        'total_explanations': ExplanationHistory.objects.filter(user=request.user).count()
    }
    
    return render(request, 'recommender/user_profile.html', context)

@login_required
def explanation_history(request):
    """Display user's explanation history"""
    explanations = ExplanationHistory.objects.filter(
        user=request.user
    ).select_related('movie').order_by('-created_at')
    
    # Pagination
    paginator = Paginator(explanations, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'explanations': page_obj,
        'total_explanations': explanations.count()
    }
    
    return render(request, 'recommender/explanation_history.html', context)

@login_required
@csrf_exempt
def rate_explanation(request):
    """Rate the helpfulness of an explanation"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            explanation_id = data.get('explanation_id')
            rating = data.get('rating')
            
            if not explanation_id or not rating:
                return JsonResponse({'success': False, 'error': 'Missing data'})
            
            if not (1 <= int(rating) <= 5):
                return JsonResponse({'success': False, 'error': 'Rating must be between 1 and 5'})
            
            explanation = get_object_or_404(
                ExplanationHistory, 
                id=explanation_id, 
                user=request.user
            )
            
            explanation.helpful_rating = int(rating)
            explanation.save()
            
            return JsonResponse({'success': True, 'message': 'Rating saved successfully'})
            
        except Exception as e:
            logger.error(f"Error rating explanation: {e}")
            return JsonResponse({'success': False, 'error': 'Server error'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def home(request):
    """Home page view"""
    if request.user.is_authenticated:
        return redirect('recommender:recommendations')
    
    # Get some popular movies for non-authenticated users
    popular_movies = Movie.objects.annotate(
        avg_rating=Avg('rating__rating')
    ).filter(
        avg_rating__isnull=False
    ).order_by('-avg_rating')[:6]
    
    context = {
        'popular_movies': popular_movies
    }
    
    return render(request, 'recommender/home.html', context)

# Additional utility views
@login_required
def delete_explanation(request, explanation_id):
    """Delete an explanation from history"""
    if request.method == 'POST':
        try:
            explanation = get_object_or_404(
                ExplanationHistory, 
                id=explanation_id, 
                user=request.user
            )
            explanation.delete()
            messages.success(request, "Explanation deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting explanation: {e}")
            messages.error(request, "Could not delete explanation.")
    
    return redirect('recommender:explanation_history')

@login_required
def clear_explanation_history(request):
    """Clear all explanation history for the user"""
    if request.method == 'POST':
        try:
            count = ExplanationHistory.objects.filter(user=request.user).count()
            ExplanationHistory.objects.filter(user=request.user).delete()
            messages.success(request, f"Cleared {count} explanations from your history.")
        except Exception as e:
            logger.error(f"Error clearing explanation history: {e}")
            messages.error(request, "Could not clear explanation history.")
    
    return redirect('recommender:explanation_history')