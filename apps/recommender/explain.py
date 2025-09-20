import numpy as np
import pandas as pd
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular
import logging
from django.core.cache import cache
from django.conf import settings
from apps.movies.models import Movie, Rating
from apps.accounts.models import User
from .engine import lightfm_engine

logger = logging.getLogger('recommender')

class RecommendationExplainer:
    """Multi-method explainer for movie recommendations"""
    
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.anchor_explainer = None
        self.feature_names = []
        self.surrogate_model = None
        
    def initialize_explainers(self):
        """Initialize all explanation methods"""
        logger.info("Initializing recommendation explainers...")
        
        # Prepare feature matrix for surrogate model
        self._prepare_surrogate_data()
        
        # Initialize SHAP explainer
        self._initialize_shap()
        
        # Initialize LIME explainer
        self._initialize_lime()
        
        # Initialize Anchor explainer
        self._initialize_anchors()
        
        logger.info("All explainers initialized successfully")
    
    def _prepare_surrogate_data(self):
        """Prepare data for training surrogate model"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Get all users and movies with ratings
        ratings_data = []
        
        for rating in Rating.objects.select_related('user', 'movie').all()[:1000]:  # Limit for performance
            user_features = self._extract_user_features(rating.user)
            movie_features = self._extract_movie_features(rating.movie)
            
            # Combine features
            combined_features = {**user_features, **movie_features}
            combined_features['rating'] = rating.rating
            
            ratings_data.append(combined_features)
        
        if not ratings_data:
            raise ValueError("No rating data available for surrogate model")
        
        # Convert to DataFrame
        df = pd.DataFrame(ratings_data)
        
        # Separate features and target
        self.feature_names = [col for col in df.columns if col != 'rating']
        X = df[self.feature_names].fillna(0)
        y = df['rating']
        
        # Train surrogate model
        self.surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.surrogate_model.fit(X, y)
        
        # Store feature matrix for explainers
        self.feature_matrix = X.values
        
        logger.info(f"Surrogate model trained with {len(self.feature_names)} features")
    
    def _extract_user_features(self, user):
        """Extract user features for explanation"""
        features = {}
        
        # Demographic features
        if user.date_of_birth:
            from datetime import date
            age = (date.today() - user.date_of_birth).days // 365
            features['user_age'] = age
        else:
            features['user_age'] = 30  # Default
        
        # Preference features
        features['user_openness'] = user.openness_to_new
        features['user_min_rating'] = user.min_rating_threshold
        
        # Genre preferences (one-hot encoded)
        all_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        for genre in all_genres:
            features[f'user_likes_{genre}'] = 1 if genre in user.favorite_genres else 0
        
        # Rating behavior
        user_ratings = Rating.objects.filter(user=user)
        features['user_avg_rating'] = user_ratings.aggregate(avg=models.Avg('rating'))['avg'] or 3.0
        features['user_total_ratings'] = user_ratings.count()
        
        return features
    
    def _extract_movie_features(self, movie):
        """Extract movie features for explanation"""
        features = {}
        
        # Basic features
        features['movie_tmdb_rating'] = movie.tmdb_rating
        features['movie_popularity'] = movie.popularity
        features['movie_runtime'] = movie.runtime or 120  # Default runtime
        
        # Release year
        if movie.release_date:
            features['movie_year'] = movie.release_date.year
        else:
            features['movie_year'] = 2000  # Default
        
        # Genre features (one-hot encoded)
        movie_genres = movie.get_genre_names()
        all_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
        for genre in all_genres:
            features[f'movie_genre_{genre}'] = 1 if genre in movie_genres else 0
        
        # Director features (simplified)
        directors = movie.get_directors()
        features['movie_has_famous_director'] = 1 if any(
            director in ['Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese', 'Quentin Tarantino']
            for director in directors
        ) else 0
        
        # Cast features (simplified)
        main_cast = movie.get_main_cast(3)
        features['movie_star_power'] = len(main_cast)  # Simple star power metric
        
        return features
    
    def _initialize_shap(self):
        """Initialize SHAP explainer"""
        if self.surrogate_model is not None:
            self.shap_explainer = shap.TreeExplainer(self.surrogate_model)
            logger.info("SHAP explainer initialized")
    
    def _initialize_lime(self):
        """Initialize LIME explainer"""
        if self.feature_matrix is not None:
            self.lime_explainer = LimeTabularExplainer(
                self.feature_matrix,
                feature_names=self.feature_names,
                mode='regression',
                random_state=42
            )
            logger.info("LIME explainer initialized")
    
    def _initialize_anchors(self):
        """Initialize Anchor explainer"""
        if self.feature_matrix is not None:
            # Convert to categorical data for anchors
            categorical_features = list(range(len(self.feature_names)))
            
            self.anchor_explainer = AnchorTabular(
                predictor=self.surrogate_model.predict,
                feature_names=self.feature_names,
                categorical_names={},  # Will be populated as needed
                seed=42
            )
            
            # Fit the explainer
            self.anchor_explainer.fit(self.feature_matrix)
            logger.info("Anchor explainer initialized")
    
    def explain_recommendation(self, user_id, movie_id):
        """Generate comprehensive explanation for a recommendation"""
        cache_key = f"explanation_{user_id}_{movie_id}"
        cached_explanation = cache.get(cache_key)
        
        if cached_explanation:
            return cached_explanation
        
        try:
            user = User.objects.get(id=user_id)
            movie = Movie.objects.get(id=movie_id)
        except (User.DoesNotExist, Movie.DoesNotExist):
            return {"error": "User or movie not found"}
        
        # Prepare feature vector for this user-movie pair
        user_features = self._extract_user_features(user)
        movie_features = self._extract_movie_features(movie)
        combined_features = {**user_features, **movie_features}
        
        # Convert to array in correct order
        feature_vector = np.array([combined_features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        
        explanation = {
            'user_id': user_id,
            'movie_id': movie_id,
            'movie_title': movie.title,
            'prediction_score': float(self.surrogate_model.predict(feature_vector)[0]),
        }
        
        # Generate SHAP explanation
        if self.shap_explainer:
            explanation['shap'] = self._generate_shap_explanation(feature_vector)
        
        # Generate LIME explanation
        if self.lime_explainer:
            explanation['lime'] = self._generate_lime_explanation(feature_vector)
        
        # Generate Anchor explanation
        if self.anchor_explainer:
            explanation['anchor'] = self._generate_anchor_explanation(feature_vector)
        
        # Generate counterfactual explanation
        explanation['counterfactual'] = self._generate_counterfactual_explanation(user, movie, combined_features)
        
        # Cache the explanation
        cache.set(cache_key, explanation, timeout=settings.EXPLANATION_CACHE_TIMEOUT)
        
        return explanation
    
    def _generate_shap_explanation(self, feature_vector):
        """Generate SHAP explanation"""
        try:
            shap_values = self.shap_explainer.shap_values(feature_vector)
            
            # Get top contributing features
            feature_importance = list(zip(self.feature_names, shap_values[0]))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return {
                'type': 'shap',
                'feature_importance': [
                    {'feature': name, 'importance': float(importance)}
                    for name, importance in feature_importance[:10]
                ],
                'explanation_text': self._format_shap_explanation(feature_importance[:5])
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'type': 'shap', 'error': str(e)}
    
    def _generate_lime_explanation(self, feature_vector):
        """Generate LIME explanation"""
        try:
            explanation = self.lime_explainer.explain_instance(
                feature_vector[0],
                self.surrogate_model.predict,
                num_features=10
            )
            
            # Extract feature importance
            feature_importance = explanation.as_list()
            
            return {
                'type': 'lime',
                'feature_importance': [
                    {'feature': feature, 'importance': float(importance)}
                    for feature, importance in feature_importance
                ],
                'explanation_text': self._format_lime_explanation(feature_importance[:5])
            }
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'type': 'lime', 'error': str(e)}
    
    def _generate_anchor_explanation(self, feature_vector):
        """Generate Anchor explanation"""
        try:
            explanation = self.anchor_explainer.explain(feature_vector[0])
            
            return {
                'type': 'anchor',
                'rules': explanation.anchor,
                'precision': float(explanation.precision),
                'coverage': float(explanation.coverage),
                'explanation_text': f"Rule: {' AND '.join(explanation.anchor)}"
            }
        except Exception as e:
            logger.error(f"Anchor explanation failed: {e}")
            return {'type': 'anchor', 'error': str(e)}
    
    def _generate_counterfactual_explanation(self, user, movie, features):
        """Generate counterfactual explanation"""
        try:
            # Simple rule-based counterfactuals
            counterfactuals = []
            
            # Genre-based counterfactuals
            movie_genres = movie.get_genre_names()
            user_favorite_genres = user.favorite_genres or []
            
            if not any(genre in user_favorite_genres for genre in movie_genres):
                missing_genres = [g for g in movie_genres if g not in user_favorite_genres]
                if missing_genres:
                    counterfactuals.append(
                        f"If you rated more {missing_genres[0]} movies highly, this recommendation would be stronger."
                    )
            
            # Rating threshold counterfactual
            if movie.tmdb_rating < user.min_rating_threshold:
                counterfactuals.append(
                    f"If you lowered your minimum rating threshold from {user.min_rating_threshold} to {movie.tmdb_rating:.1f}, you'd see more movies like this."
                )
            
            # Openness counterfactual
            if user.openness_to_new < 7 and movie.release_date and movie.release_date.year > 2020:
                counterfactuals.append(
                    "If you were more open to newer releases, you'd see more recent movies like this."
                )
            
            return {
                'type': 'counterfactual',
                'scenarios': counterfactuals[:3],  # Limit to top 3
                'explanation_text': counterfactuals[0] if counterfactuals else "No clear counterfactuals found."
            }
        except Exception as e:
            logger.error(f"Counterfactual explanation failed: {e}")
            return {'type': 'counterfactual', 'error': str(e)}
    
    def _format_shap_explanation(self, feature_importance):
        """Format SHAP explanation as natural language"""
        positive_features = [(name, imp) for name, imp in feature_importance if imp > 0]
        negative_features = [(name, imp) for name, imp in feature_importance if imp < 0]
        
        explanation = "This recommendation is based on: "
        
        if positive_features:
            pos_text = ", ".join([self._humanize_feature(name) for name, _ in positive_features[:3]])
            explanation += f"{pos_text}"
        
        if negative_features:
            neg_text = ", ".join([self._humanize_feature(name) for name, _ in negative_features[:2]])
            explanation += f", but less influenced by {neg_text}"
        
        return explanation
    
    def _format_lime_explanation(self, feature_importance):
        """Format LIME explanation as natural language"""
        return self._format_shap_explanation(feature_importance)  # Similar formatting
    
    def _humanize_feature(self, feature_name):
        """Convert feature names to human-readable text"""
        humanized = {
            'user_age': 'your age',
            'user_openness': 'your openness to new movies',
            'movie_tmdb_rating': 'the movie\'s high rating',
            'movie_popularity': 'the movie\'s popularity',
            'movie_genre_Action': 'action genre',
            'movie_genre_Comedy': 'comedy genre',
            'movie_genre_Drama': 'drama genre',
            'movie_genre_Horror': 'horror genre',
            'movie_genre_Romance': 'romance genre',
            'movie_genre_Sci-Fi': 'sci-fi genre',
            'movie_has_famous_director': 'famous director',
            'user_likes_Action': 'your love for action',
            'user_likes_Comedy': 'your love for comedy',
            'user_likes_Drama': 'your love for drama',
        }
        
        return humanized.get(feature_name, feature_name.replace('_', ' '))

# Global instance
explainer = RecommendationExplainer()