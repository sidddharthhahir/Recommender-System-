# # import numpy as np
# # import pandas as pd
# # from lightfm import LightFM
# # from lightfm.data import Dataset
# # from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
# # from sklearn.model_selection import train_test_split
# # import pickle
# # import logging
# # from django.conf import settings
# # from apps.movies.models import Movie, Rating, Genre
# # from django.contrib.auth import get_user_model
# # from apps.accounts.models import Profile
# # from datetime import date

# # logger = logging.getLogger('recommender')
# # User = get_user_model()

# # class LightFMEngine:
# #     """LightFM recommendation engine with hybrid collaborative + content filtering"""
    
# #     def __init__(self):
# #         self.model = None
# #         self.dataset = None
# #         self.user_features = None
# #         self.item_features = None
# #         self.user_id_map = {}
# #         self.item_id_map = {}
# #         self.reverse_user_map = {}
# #         self.reverse_item_map = {}
        
# #     def prepare_data(self):
# #         """Prepare data for LightFM training"""
# #         logger.info("Preparing data for LightFM training...")
        
# #         # Get all ratings
# #         ratings_df = pd.DataFrame(
# #             Rating.objects.select_related('user', 'movie').values(
# #                 'user_id', 'movie_id', 'rating', 'user__username', 'movie__title'
# #             )
# #         )
        
# #         if ratings_df.empty:
# #             raise ValueError("No ratings found in database")
        
# #         # Get movie features (genres, cast, crew, etc.)
# #         movies_df = pd.DataFrame(
# #             Movie.objects.prefetch_related('genres').values(
# #                 'id', 'title', 'tmdb_rating', 'popularity', 'release_date',
# #                 'cast', 'crew', 'keywords'
# #             )
# #         )
        
# #         # Get genre mapping
# #         movie_genres = {
# #             movie.id: list(movie.genres.values_list('name', flat=True))
# #             for movie in Movie.objects.prefetch_related('genres').all()
# #         }
        
# #         # Create dataset
# #         self.dataset = Dataset()
        
# #         unique_users = ratings_df['user_id'].unique()
# #         unique_items = ratings_df['movie_id'].unique()
        
# #         self.dataset.fit(
# #             users=unique_users,
# #             items=unique_items,
# #             user_features=self._extract_user_features(unique_users),
# #             item_features=self._extract_item_features(unique_items, movies_df, movie_genres)
# #         )
        
# #         # Create ID maps
# #         self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
# #         self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
# #         self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
# #         self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
# #         # Build interactions matrix
# #         interactions, weights = self.dataset.build_interactions(
# #             [(row['user_id'], row['movie_id'], row['rating']) for _, row in ratings_df.iterrows()]
# #         )
        
# #         # Build feature matrices
# #         self.user_features = self.dataset.build_user_features(
# #             [(uid, self._get_user_feature_list(uid)) for uid in unique_users]
# #         )
        
# #         self.item_features = self.dataset.build_item_features(
# #             [(iid, self._get_item_feature_list(iid, movies_df, movie_genres)) for iid in unique_items]
# #         )
        
# #         logger.info(f"Data prepared: {len(unique_users)} users, {len(unique_items)} items, {len(ratings_df)} interactions")
        
# #         return interactions, weights
    
# #     def _extract_user_features(self, user_ids):
# #         """Collect the global set of possible user features"""
# #         features = set()
        
# #         for user_id in user_ids:
# #             try:
# #                 user = User.objects.get(id=user_id)
# #                 if hasattr(user, "profile"):
# #                     profile = user.profile
                    
# #                     # Age group
# #                     if profile.birth_date:
# #                         age_group = self._get_age_group(profile.birth_date)
# #                         features.add(f"age_{age_group}")
                    
# #                     # Favorite genres
# #                     for genre in profile.favorite_genres or []:
# #                         features.add(f"likes_{genre}")
                    
# #                     # Openness (if exists)
# #                     if hasattr(profile, "openness_to_new") and profile.openness_to_new is not None:
# #                         features.add(f"openness_{profile.openness_to_new // 2}")
                        
# #             except User.DoesNotExist:
# #                 continue
        
# #         return list(features)
    
# #     def _extract_item_features(self, item_ids, movies_df, movie_genres):
# #         """Collect the global set of possible item features"""
# #         features = set()
        
# #         for item_id in item_ids:
# #             # Genres
# #             for genre in movie_genres.get(item_id, []):
# #                 features.add(f"genre_{genre}")
            
# #             # Movie-specific
# #             movie_data = movies_df[movies_df['id'] == item_id]
# #             if not movie_data.empty:
# #                 movie = movie_data.iloc[0]
                
# #                 # Rating tier
# #                 if movie['tmdb_rating'] >= 8.0:
# #                     features.add("rating_excellent")
# #                 elif movie['tmdb_rating'] >= 7.0:
# #                     features.add("rating_good")
# #                 elif movie['tmdb_rating'] >= 6.0:
# #                     features.add("rating_average")
# #                 else:
# #                     features.add("rating_poor")
                
# #                 # Popularity
# #                 if movie['popularity'] >= 50:
# #                     features.add("popularity_high")
# #                 elif movie['popularity'] >= 20:
# #                     features.add("popularity_medium")
# #                 else:
# #                     features.add("popularity_low")
                
# #                 # Decade
# #                 if movie['release_date']:
# #                     year = pd.to_datetime(movie['release_date']).year
# #                     decade = (year // 10) * 10
# #                     features.add(f"decade_{decade}s")
                
# #                 # Directors
# #                 if movie['crew']:
# #                     directors = [p['name'] for p in movie['crew'] if p.get('job') == 'Director']
# #                     for d in directors[:2]:
# #                         features.add(f"director_{d.replace(' ', '_')}")
        
# #         return list(features)
    
# #     def _get_user_feature_list(self, user_id):
# #         """Return features for a specific user"""
# #         features = []
        
# #         try:
# #             user = User.objects.get(id=user_id)
# #             if hasattr(user, "profile"):
# #                 profile = user.profile
                
# #                 # Age group
# #                 if profile.birth_date:
# #                     age_group = self._get_age_group(profile.birth_date)
# #                     features.append(f"age_{age_group}")
                
# #                 # Favorite genres
# #                 for genre in profile.favorite_genres or []:
# #                     features.append(f"likes_{genre}")
                
# #                 # Openness
# #                 if hasattr(profile, "openness_to_new") and profile.openness_to_new is not None:
# #                     features.append(f"openness_{profile.openness_to_new // 2}")
                    
# #         except User.DoesNotExist:
# #             pass
        
# #         return features
    
# #     def _get_item_feature_list(self, item_id, movies_df, movie_genres):
# #         """Return features for a specific movie"""
# #         features = []
        
# #         # Genres
# #         for genre in movie_genres.get(item_id, []):
# #             features.append(f"genre_{genre}")
        
# #         # Extra movie features
# #         movie_data = movies_df[movies_df['id'] == item_id]
# #         if not movie_data.empty:
# #             movie = movie_data.iloc[0]
            
# #             if movie['tmdb_rating'] >= 8.0:
# #                 features.append("rating_excellent")
# #             elif movie['tmdb_rating'] >= 7.0:
# #                 features.append("rating_good")
# #             elif movie['tmdb_rating'] >= 6.0:
# #                 features.append("rating_average")
# #             else:
# #                 features.append("rating_poor")
            
# #             if movie['popularity'] >= 50:
# #                 features.append("popularity_high")
# #             elif movie['popularity'] >= 20:
# #                 features.append("popularity_medium")
# #             else:
# #                 features.append("popularity_low")
            
# #             if movie['release_date']:
# #                 year = pd.to_datetime(movie['release_date']).year
# #                 decade = (year // 10) * 10
# #                 features.append(f"decade_{decade}s")
            
# #             if movie['crew']:
# #                 directors = [p['name'] for p in movie['crew'] if p.get('job') == 'Director']
# #                 for d in directors[:2]:
# #                     features.append(f"director_{d.replace(' ', '_')}")
        
# #         return features
    
# #     def _get_age_group(self, birth_date):
# #         """Convert birth_date into age group"""
# #         age = (date.today() - birth_date).days // 365
# #         if age < 25:
# #             return "young"
# #         elif age < 35:
# #             return "adult"
# #         elif age < 50:
# #             return "middle"
# #         else:
# #             return "senior"
    
# #     def train(self, epochs=20, num_components=50, loss='warp', learning_rate=0.05):
# #         """Train the LightFM model"""
# #         logger.info(f"Training LightFM model with {epochs} epochs...")
        
# #         interactions, weights = self.prepare_data()
        
# #         # Handle tiny datasets (< 5 interactions)
# #         if interactions.shape[0] < 5:
# #             logger.warning("Very small dataset detected. Using all data for training (no test split).")
# #             train_interactions = interactions
# #             test_interactions = interactions
# #         else:
# #             train_interactions, test_interactions = train_test_split(
# #                 interactions.tocoo(), test_size=0.2, random_state=42
# #             )
        
# #         self.model = LightFM(
# #             loss=loss,
# #             no_components=num_components,
# #             learning_rate=learning_rate,
# #             random_state=42
# #         )
        
# #         self.model.fit(
# #             interactions=train_interactions.tocsr(),
# #             user_features=self.user_features,
# #             item_features=self.item_features,
# #             epochs=epochs,
# #             verbose=True
# #         )
        
# #         # Evaluate (handle case where train == test)
# #         try:
# #             train_precision = precision_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_precision = precision_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_recall = recall_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_recall = recall_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_auc = auc_score(self.model, train_interactions.tocsr()).mean()
# #             test_auc = auc_score(self.model, test_interactions.tocsr()).mean()
            
# #             metrics = {
# #                 'train_precision_at_10': train_precision,
# #                 'test_precision_at_10': test_precision,
# #                 'train_recall_at_10': train_recall,
# #                 'test_recall_at_10': test_recall,
# #                 'train_auc': train_auc,
# #                 'test_auc': test_auc,
# #             }
            
# #             logger.info(f"Training done. Test Precision@10={test_precision:.4f}, Recall@10={test_recall:.4f}")
            
# #         except Exception as e:
# #             logger.warning(f"Evaluation failed (tiny dataset): {e}")
# #             metrics = {"status": "trained_but_no_evaluation"}
        
# #         return metrics

# #     def train_with_stats(self, epochs=20, num_components=50, loss='warp', learning_rate=0.05):
# #         """Train the LightFM model and return both metrics and training stats"""
# #         logger.info(f"Training LightFM model with {epochs} epochs...")
        
# #         interactions, weights = self.prepare_data()
        
# #         # Handle tiny datasets (< 5 interactions)
# #         if interactions.shape[0] < 5:
# #             logger.warning("Very small dataset detected. Using all data for training (no test split).")
# #             train_interactions = interactions
# #             test_interactions = interactions
# #         else:
# #             train_interactions, test_interactions = train_test_split(
# #                 interactions.tocoo(), test_size=0.2, random_state=42
# #             )
        
# #         self.model = LightFM(
# #             loss=loss,
# #             no_components=num_components,
# #             learning_rate=learning_rate,
# #             random_state=42
# #         )
        
# #         self.model.fit(
# #             interactions=train_interactions.tocsr(),
# #             user_features=self.user_features,
# #             item_features=self.item_features,
# #             epochs=epochs,
# #             verbose=True
# #         )
        
# #         # Collect training statistics
# #         training_stats = {
# #             'num_users': len(self.user_id_map),
# #             'num_items': len(self.item_id_map),
# #             'num_interactions': interactions.nnz if hasattr(interactions, 'nnz') else len(interactions.data),
# #             'training_data_size': interactions.shape[0] * interactions.shape[1],
# #         }
        
# #         # Evaluate (handle case where train == test)
# #         try:
# #             train_precision = precision_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_precision = precision_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_recall = recall_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_recall = recall_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_auc = auc_score(self.model, train_interactions.tocsr()).mean()
# #             test_auc = auc_score(self.model, test_interactions.tocsr()).mean()
            
# #             metrics = {
# #                 'train_precision_at_10': train_precision,
# #                 'test_precision_at_10': test_precision,
# #                 'train_recall_at_10': train_recall,
# #                 'test_recall_at_10': test_recall,
# #                 'train_auc': train_auc,
# #                 'test_auc': test_auc,
# #             }
            
# #             logger.info(f"Training done. Test Precision@10={test_precision:.4f}, Recall@10={test_recall:.4f}")
            
# #         except Exception as e:
# #             logger.warning(f"Evaluation failed (tiny dataset): {e}")
# #             metrics = {"status": "trained_but_no_evaluation"}
        
# #         return metrics, training_stats
    
# #     def predict(self, user_id, item_ids=None, num_recommendations=20):
# #         """Recommend movies to a specific user"""
# #         if self.model is None:
# #             raise ValueError("Model not trained. Call train() first.")
        
# #         if user_id not in self.user_id_map:
# #             logger.warning(f"User {user_id} not in training set. Cold-start.")
# #             return self._cold_start_recommendations(user_id, num_recommendations)
        
# #         user_idx = self.user_id_map[user_id]
        
# #         if item_ids is None:
# #             item_ids = list(self.item_id_map.keys())
        
# #         valid_item_ids = [iid for iid in item_ids if iid in self.item_id_map]
# #         item_indices = [self.item_id_map[iid] for iid in valid_item_ids]
        
# #         existing = set(Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True))
        
# #         scores = self.model.predict(
# #             user_ids=[user_idx] * len(item_indices),
# #             item_ids=item_indices,
# #             user_features=self.user_features,
# #             item_features=self.item_features
# #         )
        
# #         recommendations = []
# #         for item_id, score in zip(valid_item_ids, scores):
# #             if item_id not in existing:
# #                 recommendations.append({
# #                     "movie_id": item_id,
# #                     "score": float(score)
# #                 })
        
# #         recommendations.sort(key=lambda x: x['score'], reverse=True)
# #         return recommendations[:num_recommendations]
    
# #     def _cold_start_recommendations(self, user_id, num_recommendations):
# #         """Fallback recs for users with no interactions"""
# #         try:
# #             user = User.objects.get(id=user_id)
# #             favorite_genres = []
# #             if hasattr(user, "profile") and user.profile.favorite_genres:
# #                 favorite_genres = user.profile.favorite_genres
            
# #             if favorite_genres:
# #                 movies = Movie.objects.filter(
# #                     genres__name__in=favorite_genres,
# #                     tmdb_rating__gte=7.0,
# #                     is_active=True
# #                 ).exclude(ratings__user=user).order_by("-popularity", "-tmdb_rating")[:num_recommendations]
# #             else:
# #                 movies = Movie.objects.filter(
# #                     tmdb_rating__gte=7.5,
# #                     is_active=True
# #                 ).exclude(ratings__user=user).order_by("-popularity")[:num_recommendations]
            
# #             return [{"movie_id": m.id, "score": 0.8 - i * 0.05} for i, m in enumerate(movies)]
        
# #         except User.DoesNotExist:
# #             return []
    
# #     def save_model(self, filepath):
# #         model_data = {
# #             'model': self.model,
# #             'dataset': self.dataset,
# #             'user_features': self.user_features,
# #             'item_features': self.item_features,
# #             'user_id_map': self.user_id_map,
# #             'item_id_map': self.item_id_map,
# #             'reverse_user_map': self.reverse_user_map,
# #             'reverse_item_map': self.reverse_item_map,
# #         }
# #         with open(filepath, "wb") as f:
# #             pickle.dump(model_data, f)
# #         logger.info(f"Model saved to {filepath}")
    
# #     def load_model(self, filepath):
# #         with open(filepath, "rb") as f:
# #             data = pickle.load(f)
# #         self.model = data['model']
# #         self.dataset = data['dataset']
# #         self.user_features = data['user_features']
# #         self.item_features = data['item_features']
# #         self.user_id_map = data['user_id_map']
# #         self.item_id_map = data['item_id_map']
# #         self.reverse_user_map = data['reverse_user_map']
# #         self.reverse_item_map = data['reverse_item_map']
# #         logger.info(f"Model loaded from {filepath}")

# # # Global engine instance
# # lightfm_engine = LightFMEngine()
# # import numpy as np
# # import pandas as pd
# # from lightfm import LightFM
# # from lightfm.data import Dataset
# # from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
# # from sklearn.model_selection import train_test_split
# # import pickle
# # import logging
# # from django.conf import settings
# # from apps.movies.models import Movie, Rating, Genre
# # from django.contrib.auth import get_user_model
# # from apps.accounts.models import Profile
# # from datetime import date
# # import shap
# # from lime.lime_tabular import LimeTabularExplainer

# # logger = logging.getLogger('recommender')
# # User = get_user_model()

# # class LightFMEngine:
# #     """LightFM recommendation engine with hybrid collaborative + content filtering"""
    
# #     def __init__(self):
# #         self.model = None
# #         self.dataset = None
# #         self.user_features = None
# #         self.item_features = None
# #         self.user_id_map = {}
# #         self.item_id_map = {}
# #         self.reverse_user_map = {}
# #         self.reverse_item_map = {}
        
# #     def prepare_data(self):
# #         """Prepare data for LightFM training"""
# #         logger.info("Preparing data for LightFM training...")
        
# #         # Get all ratings
# #         ratings_df = pd.DataFrame(
# #             Rating.objects.select_related('user', 'movie').values(
# #                 'user_id', 'movie_id', 'rating', 'user__username', 'movie__title'
# #             )
# #         )
        
# #         if ratings_df.empty:
# #             raise ValueError("No ratings found in database")
        
# #         # Get movie features (genres, cast, crew, etc.)
# #         movies_df = pd.DataFrame(
# #             Movie.objects.prefetch_related('genres').values(
# #                 'id', 'title', 'tmdb_rating', 'popularity', 'release_date',
# #                 'cast', 'crew', 'keywords'
# #             )
# #         )
        
# #         # Get genre mapping
# #         movie_genres = {
# #             movie.id: list(movie.genres.values_list('name', flat=True))
# #             for movie in Movie.objects.prefetch_related('genres').all()
# #         }
        
# #         # Create dataset
# #         self.dataset = Dataset()
        
# #         unique_users = ratings_df['user_id'].unique()
# #         unique_items = ratings_df['movie_id'].unique()
        
# #         self.dataset.fit(
# #             users=unique_users,
# #             items=unique_items,
# #             user_features=self._extract_user_features(unique_users),
# #             item_features=self._extract_item_features(unique_items, movies_df, movie_genres)
# #         )
        
# #         # Create ID maps
# #         self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
# #         self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
# #         self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
# #         self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
# #         # Build interactions matrix
# #         interactions, weights = self.dataset.build_interactions(
# #             [(row['user_id'], row['movie_id'], row['rating']) for _, row in ratings_df.iterrows()]
# #         )
        
# #         # Build feature matrices
# #         self.user_features = self.dataset.build_user_features(
# #             [(uid, self._get_user_feature_list(uid)) for uid in unique_users]
# #         )
        
# #         self.item_features = self.dataset.build_item_features(
# #             [(iid, self._get_item_feature_list(iid, movies_df, movie_genres)) for iid in unique_items]
# #         )
        
# #         logger.info(f"Data prepared: {len(unique_users)} users, {len(unique_items)} items, {len(ratings_df)} interactions")
        
# #         return interactions, weights
    
# #     def _extract_user_features(self, user_ids):
# #         """Collect the global set of possible user features"""
# #         features = set()
        
# #         for user_id in user_ids:
# #             try:
# #                 user = User.objects.get(id=user_id)
# #                 if hasattr(user, "profile"):
# #                     profile = user.profile
                    
# #                     # Age group
# #                     if profile.birth_date:
# #                         age_group = self._get_age_group(profile.birth_date)
# #                         features.add(f"age_{age_group}")
                    
# #                     # Favorite genres
# #                     for genre in profile.favorite_genres or []:
# #                         features.add(f"likes_{genre}")
                    
# #                     # Openness (if exists)
# #                     if hasattr(profile, "openness_to_new") and profile.openness_to_new is not None:
# #                         features.add(f"openness_{profile.openness_to_new // 2}")
                        
# #             except User.DoesNotExist:
# #                 continue
        
# #         return list(features)
    
# #     def _extract_item_features(self, item_ids, movies_df, movie_genres):
# #         """Collect the global set of possible item features"""
# #         features = set()
        
# #         for item_id in item_ids:
# #             # Genres
# #             for genre in movie_genres.get(item_id, []):
# #                 features.add(f"genre_{genre}")
            
# #             # Movie-specific
# #             movie_data = movies_df[movies_df['id'] == item_id]
# #             if not movie_data.empty:
# #                 movie = movie_data.iloc[0]
                
# #                 # Rating tier
# #                 if movie['tmdb_rating'] >= 8.0:
# #                     features.add("rating_excellent")
# #                 elif movie['tmdb_rating'] >= 7.0:
# #                     features.add("rating_good")
# #                 elif movie['tmdb_rating'] >= 6.0:
# #                     features.add("rating_average")
# #                 else:
# #                     features.add("rating_poor")
                
# #                 # Popularity
# #                 if movie['popularity'] >= 50:
# #                     features.add("popularity_high")
# #                 elif movie['popularity'] >= 20:
# #                     features.add("popularity_medium")
# #                 else:
# #                     features.add("popularity_low")
                
# #                 # Decade
# #                 if movie['release_date']:
# #                     year = pd.to_datetime(movie['release_date']).year
# #                     decade = (year // 10) * 10
# #                     features.add(f"decade_{decade}s")
                
# #                 # Directors
# #                 if movie['crew']:
# #                     directors = [p['name'] for p in movie['crew'] if p.get('job') == 'Director']
# #                     for d in directors[:2]:
# #                         features.add(f"director_{d.replace(' ', '_')}")
        
# #         return list(features)
    
# #     def _get_user_feature_list(self, user_id):
# #         """Return features for a specific user"""
# #         features = []
        
# #         try:
# #             user = User.objects.get(id=user_id)
# #             if hasattr(user, "profile"):
# #                 profile = user.profile
                
# #                 # Age group
# #                 if profile.birth_date:
# #                     age_group = self._get_age_group(profile.birth_date)
# #                     features.append(f"age_{age_group}")
                
# #                 # Favorite genres
# #                 for genre in profile.favorite_genres or []:
# #                     features.append(f"likes_{genre}")
                
# #                 # Openness
# #                 if hasattr(profile, "openness_to_new") and profile.openness_to_new is not None:
# #                     features.append(f"openness_{profile.openness_to_new // 2}")
                    
# #         except User.DoesNotExist:
# #             pass
        
# #         return features
    
# #     def _get_item_feature_list(self, item_id, movies_df, movie_genres):
# #         """Return features for a specific movie"""
# #         features = []
        
# #         # Genres
# #         for genre in movie_genres.get(item_id, []):
# #             features.append(f"genre_{genre}")
        
# #         # Extra movie features
# #         movie_data = movies_df[movies_df['id'] == item_id]
# #         if not movie_data.empty:
# #             movie = movie_data.iloc[0]
            
# #             if movie['tmdb_rating'] >= 8.0:
# #                 features.append("rating_excellent")
# #             elif movie['tmdb_rating'] >= 7.0:
# #                 features.append("rating_good")
# #             elif movie['tmdb_rating'] >= 6.0:
# #                 features.append("rating_average")
# #             else:
# #                 features.append("rating_poor")
            
# #             if movie['popularity'] >= 50:
# #                 features.append("popularity_high")
# #             elif movie['popularity'] >= 20:
# #                 features.append("popularity_medium")
# #             else:
# #                 features.append("popularity_low")
            
# #             if movie['release_date']:
# #                 year = pd.to_datetime(movie['release_date']).year
# #                 decade = (year // 10) * 10
# #                 features.append(f"decade_{decade}s")
            
# #             if movie['crew']:
# #                 directors = [p['name'] for p in movie['crew'] if p.get('job') == 'Director']
# #                 for d in directors[:2]:
# #                     features.append(f"director_{d.replace(' ', '_')}")
        
# #         return features
    
# #     def _get_age_group(self, birth_date):
# #         """Convert birth_date into age group"""
# #         age = (date.today() - birth_date).days // 365
# #         if age < 25:
# #             return "young"
# #         elif age < 35:
# #             return "adult"
# #         elif age < 50:
# #             return "middle"
# #         else:
# #             return "senior"
    
# #     def train(self, epochs=20, num_components=50, loss='warp', learning_rate=0.05):
# #         """Train the LightFM model"""
# #         logger.info(f"Training LightFM model with {epochs} epochs...")
        
# #         interactions, weights = self.prepare_data()
        
# #         # Handle tiny datasets (< 5 interactions)
# #         if interactions.shape[0] < 5:
# #             logger.warning("Very small dataset detected. Using all data for training (no test split).")
# #             train_interactions = interactions
# #             test_interactions = interactions
# #         else:
# #             train_interactions, test_interactions = train_test_split(
# #                 interactions.tocoo(), test_size=0.2, random_state=42
# #             )
        
# #         self.model = LightFM(
# #             loss=loss,
# #             no_components=num_components,
# #             learning_rate=learning_rate,
# #             random_state=42
# #         )
        
# #         self.model.fit(
# #             interactions=train_interactions.tocsr(),
# #             user_features=self.user_features,
# #             item_features=self.item_features,
# #             epochs=epochs,
# #             verbose=True
# #         )
        
# #         # Evaluate (handle case where train == test)
# #         try:
# #             train_precision = precision_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_precision = precision_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_recall = recall_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_recall = recall_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_auc = auc_score(self.model, train_interactions.tocsr()).mean()
# #             test_auc = auc_score(self.model, test_interactions.tocsr()).mean()
            
# #             metrics = {
# #                 'train_precision_at_10': train_precision,
# #                 'test_precision_at_10': test_precision,
# #                 'train_recall_at_10': train_recall,
# #                 'test_recall_at_10': test_recall,
# #                 'train_auc': train_auc,
# #                 'test_auc': test_auc,
# #             }
            
# #             logger.info(f"Training done. Test Precision@10={test_precision:.4f}, Recall@10={test_recall:.4f}")
            
# #         except Exception as e:
# #             logger.warning(f"Evaluation failed (tiny dataset): {e}")
# #             metrics = {"status": "trained_but_no_evaluation"}
        
# #         return metrics

# #     def train_with_stats(self, epochs=20, num_components=50, loss='warp', learning_rate=0.05):
# #         """Train the LightFM model and return both metrics and training stats"""
# #         logger.info(f"Training LightFM model with {epochs} epochs...")
        
# #         interactions, weights = self.prepare_data()
        
# #         # Handle tiny datasets (< 5 interactions)
# #         if interactions.shape[0] < 5:
# #             logger.warning("Very small dataset detected. Using all data for training (no test split).")
# #             train_interactions = interactions
# #             test_interactions = interactions
# #         else:
# #             train_interactions, test_interactions = train_test_split(
# #                 interactions.tocoo(), test_size=0.2, random_state=42
# #             )
        
# #         self.model = LightFM(
# #             loss=loss,
# #             no_components=num_components,
# #             learning_rate=learning_rate,
# #             random_state=42
# #         )
        
# #         self.model.fit(
# #             interactions=train_interactions.tocsr(),
# #             user_features=self.user_features,
# #             item_features=self.item_features,
# #             epochs=epochs,
# #             verbose=True
# #         )
        
# #         # Collect training statistics
# #         training_stats = {
# #             'num_users': len(self.user_id_map),
# #             'num_items': len(self.item_id_map),
# #             'num_interactions': interactions.nnz if hasattr(interactions, 'nnz') else len(interactions.data),
# #             'training_data_size': interactions.shape[0] * interactions.shape[1],
# #         }
        
# #         # Evaluate (handle case where train == test)
# #         try:
# #             train_precision = precision_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_precision = precision_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_recall = recall_at_k(self.model, train_interactions.tocsr(), k=10).mean()
# #             test_recall = recall_at_k(self.model, test_interactions.tocsr(), k=10).mean()
            
# #             train_auc = auc_score(self.model, train_interactions.tocsr()).mean()
# #             test_auc = auc_score(self.model, test_interactions.tocsr()).mean()
            
# #             metrics = {
# #                 'train_precision_at_10': train_precision,
# #                 'test_precision_at_10': test_precision,
# #                 'train_recall_at_10': train_recall,
# #                 'test_recall_at_10': test_recall,
# #                 'train_auc': train_auc,
# #                 'test_auc': test_auc,
# #             }
            
# #             logger.info(f"Training done. Test Precision@10={test_precision:.4f}, Recall@10={test_recall:.4f}")
            
# #         except Exception as e:
# #             logger.warning(f"Evaluation failed (tiny dataset): {e}")
# #             metrics = {"status": "trained_but_no_evaluation"}
        
# #         return metrics, training_stats
    
# #     def predict(self, user_id, item_ids=None, num_recommendations=20):
# #         """Recommend movies to a specific user"""
# #         if self.model is None:
# #             raise ValueError("Model not trained. Call train() first.")
        
# #         if user_id not in self.user_id_map:
# #             logger.warning(f"User {user_id} not in training set. Cold-start.")
# #             return self._cold_start_recommendations(user_id, num_recommendations)
        
# #         user_idx = self.user_id_map[user_id]
        
# #         if item_ids is None:
# #             item_ids = list(self.item_id_map.keys())
        
# #         valid_item_ids = [iid for iid in item_ids if iid in self.item_id_map]
# #         item_indices = [self.item_id_map[iid] for iid in valid_item_ids]
        
# #         existing = set(Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True))
        
# #         scores = self.model.predict(
# #             user_ids=[user_idx] * len(item_indices),
# #             item_ids=item_indices,
# #             user_features=self.user_features,
# #             item_features=self.item_features
# #         )
        
# #         recommendations = []
# #         for item_id, score in zip(valid_item_ids, scores):
# #             if item_id not in existing:
# #                 recommendations.append({
# #                     "movie_id": item_id,
# #                     "score": float(score)
# #                 })
        
# #         recommendations.sort(key=lambda x: x['score'], reverse=True)
# #         return recommendations[:num_recommendations]
    
# #     def _cold_start_recommendations(self, user_id, num_recommendations):
# #         """Fallback recs for users with no interactions"""
# #         try:
# #             user = User.objects.get(id=user_id)
# #             favorite_genres = []
# #             if hasattr(user, "profile") and user.profile.favorite_genres:
# #                 favorite_genres = user.profile.favorite_genres
            
# #             if favorite_genres:
# #                 movies = Movie.objects.filter(
# #                     genres__name__in=favorite_genres,
# #                     tmdb_rating__gte=7.0,
# #                     is_active=True
# #                 ).exclude(ratings__user=user).order_by("-popularity", "-tmdb_rating")[:num_recommendations]
# #             else:
# #                 movies = Movie.objects.filter(
# #                     tmdb_rating__gte=7.5,
# #                     is_active=True
# #                 ).exclude(ratings__user=user).order_by("-popularity")[:num_recommendations]
            
# #             return [{"movie_id": m.id, "score": 0.8 - i * 0.05} for i, m in enumerate(movies)]
        
# #         except User.DoesNotExist:
# #             return []
    
# #     def save_model(self, filepath):
# #         model_data = {
# #             'model': self.model,
# #             'dataset': self.dataset,
# #             'user_features': self.user_features,
# #             'item_features': self.item_features,
# #             'user_id_map': self.user_id_map,
# #             'item_id_map': self.item_id_map,
# #             'reverse_user_map': self.reverse_user_map,
# #             'reverse_item_map': self.reverse_item_map,
# #         }
# #         with open(filepath, "wb") as f:
# #             pickle.dump(model_data, f)
# #         logger.info(f"Model saved to {filepath}")
    
# #     def load_model(self, filepath):
# #         with open(filepath, "rb") as f:
# #             data = pickle.load(f)
# #         self.model = data['model']
# #         self.dataset = data['dataset']
# #         self.user_features = data['user_features']
# #         self.item_features = data['item_features']
# #         self.user_id_map = data['user_id_map']
# #         self.item_id_map = data['item_id_map']
# #         self.reverse_user_map = data['reverse_user_map']
# #         self.reverse_item_map = data['reverse_item_map']
# #         logger.info(f"Model loaded from {filepath}")


# # class ExplanationEngine:
# #     """Generate SHAP and LIME explanations for recommendations"""

# #     def __init__(self, lightfm_engine):
# #         self.engine = lightfm_engine
# #         self.shap_explainer = None
# #         self.lime_explainer = None
# #         self.is_initialized = False

# #     def init_explainers(self):
# #         """Initialize SHAP and LIME explainers after model training"""
# #         if self.engine.model is None:
# #             logger.error("Cannot initialize explainers: LightFM model not trained")
# #             return False

# #         try:
# #             logger.info("Initializing SHAP explainer...")
            
# #             # Create prediction function for SHAP
# #             def predict_fn(user_item_pairs):
# #                 """Prediction function that takes (user_idx, item_idx) pairs"""
# #                 if len(user_item_pairs.shape) == 1:
# #                     user_item_pairs = user_item_pairs.reshape(1, -1)
                
# #                 users = user_item_pairs[:, 0].astype(int)
# #                 items = user_item_pairs[:, 1].astype(int)
                
# #                 return self.engine.model.predict(
# #                     users, items,
# #                     user_features=self.engine.user_features,
# #                     item_features=self.engine.item_features
# #                 )

# #             # Create background dataset (sample of user-item pairs)
# #             user_ids = list(self.engine.user_id_map.values())[:5]  # First 5 users
# #             item_ids = list(self.engine.item_id_map.values())[:10]  # First 10 items
            
# #             background = np.array([[u, i] for u in user_ids for i in item_ids])
            
# #             self.shap_explainer = shap.KernelExplainer(predict_fn, background[:20])  # Limit background size
            
# #             logger.info("Initializing LIME explainer...")
            
# #             # LIME explainer for item features
# #             if self.engine.item_features is not None:
# #                 # Convert sparse matrix to dense for LIME
# #                 item_features_dense = self.engine.item_features.toarray()
                
# #                 self.lime_explainer = LimeTabularExplainer(
# #                     training_data=item_features_dense,
# #                     feature_names=[f"feature_{i}" for i in range(item_features_dense.shape[1])],
# #                     mode='regression',
# #                     verbose=False
# #                 )
            
# #             self.is_initialized = True
# #             logger.info("✅ SHAP and LIME explainers initialized successfully")
# #             return True
            
# #         except Exception as e:
# #             logger.error(f"Failed to initialize explainers: {e}")
# #             return False

# #     def explain_shap(self, user_id, item_id, max_evals=50):
# #         """Generate SHAP explanation for a user-item prediction"""
# #         if not self.is_initialized:
# #             return {"error": "Explainers not initialized"}
        
# #         try:
# #             user_idx = self.engine.user_id_map.get(user_id)
# #             item_idx = self.engine.item_id_map.get(item_id)
            
# #             if user_idx is None or item_idx is None:
# #                 return {"error": "User or item not found in training data"}
            
# #             # Create instance to explain
# #             instance = np.array([[user_idx, item_idx]])
            
# #             # Generate SHAP values
# #             shap_values = self.shap_explainer.shap_values(instance, nsamples=max_evals)
            
# #             return {
# #                 "shap_values": shap_values.tolist(),
# #                 "base_value": float(self.shap_explainer.expected_value),
# #                 "prediction": float(self.engine.model.predict([user_idx], [item_idx], 
# #                                                             user_features=self.engine.user_features,
# #                                                             item_features=self.engine.item_features)[0])
# #             }
            
# #         except Exception as e:
# #             logger.error(f"SHAP explanation failed: {e}")
# #             return {"error": str(e)}

# #     def explain_lime(self, user_id, item_id, num_features=5):
# #         """Generate LIME explanation for a user-item prediction"""
# #         if not self.is_initialized or self.lime_explainer is None:
# #             return {"error": "LIME explainer not initialized"}
        
# #         try:
# #             user_idx = self.engine.user_id_map.get(user_id)
# #             item_idx = self.engine.item_id_map.get(item_id)
            
# #             if user_idx is None or item_idx is None:
# #                 return {"error": "User or item not found in training data"}
            
# #             # Get item features for this item
# #             item_features = self.engine.item_features[item_idx].toarray().flatten()
            
# #             # Create prediction function for LIME
# #             def predict_fn(item_feature_matrix):
# #                 """Predict for multiple item feature vectors with fixed user"""
# #                 predictions = []
# #                 for features in item_feature_matrix:
# #                     # This is a simplified approach - in practice you'd need to 
# #                     # reconstruct the full feature matrix
# #                     pred = self.engine.model.predict([user_idx], [item_idx],
# #                                                    user_features=self.engine.user_features,
# #                                                    item_features=self.engine.item_features)[0]
# #                     predictions.append(pred)
# #                 return np.array(predictions)
            
# #             # Generate LIME explanation
# #             explanation = self.lime_explainer.explain_instance(
# #                 item_features, predict_fn, num_features=num_features
# #             )
            
# #             return {
# #                 "lime_explanation": explanation.as_list(),
# #                 "prediction": float(self.engine.model.predict([user_idx], [item_idx],
# #                                                             user_features=self.engine.user_features,
# #                                                             item_features=self.engine.item_features)[0])
# #             }
            
# #         except Exception as e:
# #             logger.error(f"LIME explanation failed: {e}")
# #             return {"error": str(e)}

# #     def generate_feature_importance(self, user_id, item_id):
# #         """Generate a combined feature importance explanation"""
# #         shap_result = self.explain_shap(user_id, item_id)
# #         lime_result = self.explain_lime(user_id, item_id)
        
# #         return {
# #             "shap": shap_result,
# #             "lime": lime_result,
# #             "user_id": user_id,
# #             "item_id": item_id
# #         }


# # # Global engine instances
# # lightfm_engine = LightFMEngine()
# # explanation_engine = ExplanationEngine(lightfm_engine)
# import numpy as np
# import pandas as pd
# from lightfm import LightFM
# from lightfm.data import Dataset
# from scipy.sparse import csr_matrix
# import pickle
# import os
# from django.conf import settings
# from .models import Movie, Rating, UserProfile
# from django.contrib.auth.models import User
# import logging

# logger = logging.getLogger(__name__)

# class LightFMEngine:
#     def __init__(self):
#         self.model = None
#         self.dataset = None
#         self.user_features = None
#         self.item_features = None
#         self.user_id_map = {}
#         self.item_id_map = {}
#         self.reverse_user_id_map = {}
#         self.reverse_item_id_map = {}
#         self.feature_names = []
        
#     def prepare_data(self):
#         """Prepare data for LightFM training"""
#         try:
#             # Get all ratings
#             ratings = Rating.objects.select_related('user', 'movie').all()
#             if not ratings.exists():
#                 logger.warning("No ratings found for training")
#                 return False
                
#             # Create interaction matrix
#             interactions = []
#             for rating in ratings:
#                 interactions.append({
#                     'user_id': rating.user.id,
#                     'item_id': rating.movie.id,
#                     'rating': rating.rating
#                 })
            
#             df = pd.DataFrame(interactions)
            
#             # Create dataset
#             self.dataset = Dataset()
            
#             # Get user features (genres from user profiles)
#             user_features = []
#             for user in User.objects.all():
#                 try:
#                     profile = user.userprofile
#                     genres = [badge.name for badge in profile.badges.all()]
#                     user_features.append((user.id, genres))
#                 except:
#                     user_features.append((user.id, []))
            
#             # Get item features (movie genres)
#             item_features = []
#             for movie in Movie.objects.all():
#                 genres = movie.genres.split('|') if movie.genres else []
#                 item_features.append((movie.id, genres))
            
#             # Fit the dataset
#             self.dataset.fit(
#                 users=df['user_id'].unique(),
#                 items=df['item_id'].unique(),
#                 user_features=[genre for _, genres in user_features for genre in genres],
#                 item_features=[genre for _, genres in item_features for genre in genres]
#             )
            
#             # Store feature names for explanations
#             self.feature_names = list(set([genre for _, genres in user_features + item_features for genre in genres]))
            
#             # Build interactions matrix
#             interactions_matrix, weights = self.dataset.build_interactions(
#                 [(row['user_id'], row['item_id'], row['rating']) for _, row in df.iterrows()]
#             )
            
#             # Build feature matrices
#             user_features_matrix = self.dataset.build_user_features(user_features)
#             item_features_matrix = self.dataset.build_item_features(item_features)
            
#             self.user_features = user_features_matrix
#             self.item_features = item_features_matrix
            
#             # Store mappings
#             self.user_id_map = self.dataset.mapping()[0]
#             self.item_id_map = self.dataset.mapping()[2]
#             self.reverse_user_id_map = {v: k for k, v in self.user_id_map.items()}
#             self.reverse_item_id_map = {v: k for k, v in self.item_id_map.items()}
            
#             return interactions_matrix, weights
            
#         except Exception as e:
#             logger.error(f"Error preparing data: {e}")
#             return False
    
#     def train(self, epochs=50, num_threads=2):
#         """Train the LightFM model"""
#         try:
#             data = self.prepare_data()
#             if not data:
#                 return False
                
#             interactions_matrix, weights = data
            
#             # Initialize model
#             self.model = LightFM(
#                 loss='warp',
#                 learning_rate=0.05,
#                 item_alpha=1e-6,
#                 user_alpha=1e-6,
#                 no_components=50
#             )
            
#             # Train model
#             self.model.fit(
#                 interactions_matrix,
#                 user_features=self.user_features,
#                 item_features=self.item_features,
#                 epochs=epochs,
#                 num_threads=num_threads,
#                 verbose=True
#             )
            
#             logger.info("Model training completed successfully")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error training model: {e}")
#             return False
    
#     def predict(self, user_id, num_recommendations=10):
#         """Generate recommendations for a user"""
#         try:
#             if not self.model or user_id not in self.user_id_map:
#                 return []
            
#             internal_user_id = self.user_id_map[user_id]
            
#             # Get all items
#             all_items = list(self.item_id_map.keys())
#             internal_item_ids = [self.item_id_map[item_id] for item_id in all_items]
            
#             # Get predictions
#             scores = self.model.predict(
#                 internal_user_id,
#                 internal_item_ids,
#                 user_features=self.user_features,
#                 item_features=self.item_features,
#                 num_threads=2
#             )
            
#             # Get user's already rated movies to exclude them
#             user_rated_movies = set(
#                 Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True)
#             )
            
#             # Create recommendations with scores
#             recommendations = []
#             for i, score in enumerate(scores):
#                 movie_id = all_items[i]
#                 if movie_id not in user_rated_movies:
#                     try:
#                         movie = Movie.objects.get(id=movie_id)
#                         recommendations.append({
#                             'movie': movie,
#                             'score': float(score),
#                             'confidence_score': round(float(score), 3)
#                         })
#                     except Movie.DoesNotExist:
#                         continue
            
#             # Sort by score and return top N (remove duplicates)
#             recommendations.sort(key=lambda x: x['score'], reverse=True)
            
#             # Remove duplicates based on movie ID
#             seen_movies = set()
#             unique_recommendations = []
#             for rec in recommendations:
#                 if rec['movie'].id not in seen_movies:
#                     seen_movies.add(rec['movie'].id)
#                     unique_recommendations.append(rec)
            
#             return unique_recommendations[:num_recommendations]
            
#         except Exception as e:
#             logger.error(f"Error generating predictions: {e}")
#             return []
    
#     def explain(self, user_id, movie_id):
#         """Generate explanation for why a movie was recommended"""
#         try:
#             if not self.model or user_id not in self.user_id_map or movie_id not in self.item_id_map:
#                 return {"error": "Model not trained or user/movie not found"}
            
#             internal_user_id = self.user_id_map[user_id]
#             internal_movie_id = self.item_id_map[movie_id]
            
#             # Get user and item embeddings
#             user_embedding = self.model.user_embeddings[internal_user_id]
#             item_embedding = self.model.item_embeddings[internal_movie_id]
            
#             # Get user and item biases
#             user_bias = self.model.user_biases[internal_user_id] if hasattr(self.model, 'user_biases') else 0
#             item_bias = self.model.item_biases[internal_movie_id] if hasattr(self.model, 'item_biases') else 0
            
#             # Calculate feature contributions (simplified approach)
#             feature_contributions = user_embedding * item_embedding
            
#             # Get movie and user info for explanation
#             movie = Movie.objects.get(id=movie_id)
#             user = User.objects.get(id=user_id)
            
#             # Get user's preferred genres from profile
#             try:
#                 user_profile = user.userprofile
#                 user_genres = [badge.name for badge in user_profile.badges.all()]
#             except:
#                 user_genres = []
            
#             # Get movie genres
#             movie_genres = movie.genres.split('|') if movie.genres else []
            
#             # Find matching genres
#             matching_genres = list(set(user_genres) & set(movie_genres))
            
#             # Create explanation
#             explanation = {
#                 "prediction_score": round(float(np.dot(user_embedding, item_embedding) + user_bias + item_bias), 3),
#                 "user_bias": round(float(user_bias), 3),
#                 "item_bias": round(float(item_bias), 3),
#                 "matching_genres": matching_genres if matching_genres else ["No direct genre matches"],
#                 "movie_genres": movie_genres,
#                 "user_preferred_genres": user_genres,
#                 "top_feature_contributions": [
#                     f"Feature {i}: {round(float(contrib), 3)}" 
#                     for i, contrib in enumerate(feature_contributions[:5])
#                 ]
#             }
            
#             # Add human-readable explanation
#             if matching_genres:
#                 explanation["reason"] = f"Recommended because you like {', '.join(matching_genres)} movies"
#             elif movie.tmdb_rating and float(movie.tmdb_rating) > 7.0:
#                 explanation["reason"] = f"Recommended because it's a highly rated movie ({movie.tmdb_rating}/10)"
#             else:
#                 explanation["reason"] = "Recommended based on your viewing patterns and preferences"
            
#             return explanation
            
#         except Exception as e:
#             logger.error(f"Error generating explanation: {e}")
#             return {"error": f"Could not generate explanation: {str(e)}"}
    
#     def save_model(self, filepath):
#         """Save the trained model"""
#         try:
#             model_data = {
#                 'model': self.model,
#                 'dataset': self.dataset,
#                 'user_features': self.user_features,
#                 'item_features': self.item_features,
#                 'user_id_map': self.user_id_map,
#                 'item_id_map': self.item_id_map,
#                 'reverse_user_id_map': self.reverse_user_id_map,
#                 'reverse_item_id_map': self.reverse_item_id_map,
#                 'feature_names': self.feature_names
#             }
            
#             with open(filepath, 'wb') as f:
#                 pickle.dump(model_data, f)
            
#             logger.info(f"Model saved to {filepath}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error saving model: {e}")
#             return False
    
#     def load_model(self, filepath):
#         """Load a trained model"""
#         try:
#             if not os.path.exists(filepath):
#                 logger.warning(f"Model file not found: {filepath}")
#                 return False
            
#             with open(filepath, 'rb') as f:
#                 model_data = pickle.load(f)
            
#             self.model = model_data['model']
#             self.dataset = model_data['dataset']
#             self.user_features = model_data['user_features']
#             self.item_features = model_data['item_features']
#             self.user_id_map = model_data['user_id_map']
#             self.item_id_map = model_data['item_id_map']
#             self.reverse_user_id_map = model_data['reverse_user_id_map']
#             self.reverse_item_id_map = model_data['reverse_item_id_map']
#             self.feature_names = model_data.get('feature_names', [])
            
#             logger.info(f"Model loaded from {filepath}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error loading model: {e}")
#             return False

# # Global engine instance
# engine = LightFMEngine()
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
import pickle
import os
from django.conf import settings
from django.contrib.auth.models import User
import logging

logger = logging.getLogger(__name__)

class LightFMEngine:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.user_features = None
        self.item_features = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_id_map = {}
        self.reverse_item_id_map = {}
        self.feature_names = []
        
    def prepare_data(self):
        """Prepare data for LightFM training"""
        try:
            # Import models here to avoid circular imports
            from .models import Movie, Rating, UserProfile
            
            # Get all ratings
            ratings = Rating.objects.select_related('user', 'movie').all()
            if not ratings.exists():
                logger.warning("No ratings found for training")
                return False
                
            # Create interaction matrix
            interactions = []
            for rating in ratings:
                interactions.append({
                    'user_id': rating.user.id,
                    'item_id': rating.movie.id,
                    'rating': rating.rating
                })
            
            df = pd.DataFrame(interactions)
            
            # Create dataset
            self.dataset = Dataset()
            
            # Get user features (genres from user profiles)
            user_features = []
            for user in User.objects.all():
                try:
                    profile = user.userprofile
                    if profile.preferred_genres:
                        genres = [genre.strip() for genre in profile.preferred_genres.split(',')]
                    else:
                        genres = []
                    user_features.append((user.id, genres))
                except:
                    user_features.append((user.id, []))
            
            # Get item features (movie genres)
            item_features = []
            for movie in Movie.objects.all():
                genres = movie.genres.split('|') if movie.genres else []
                item_features.append((movie.id, genres))
            
            # Fit the dataset
            self.dataset.fit(
                users=df['user_id'].unique(),
                items=df['item_id'].unique(),
                user_features=[genre for _, genres in user_features for genre in genres],
                item_features=[genre for _, genres in item_features for genre in genres]
            )
            
            # Store feature names for explanations
            self.feature_names = list(set([genre for _, genres in user_features + item_features for genre in genres]))
            
            # Build interactions matrix
            interactions_matrix, weights = self.dataset.build_interactions(
                [(row['user_id'], row['item_id'], row['rating']) for _, row in df.iterrows()]
            )
            
            # Build feature matrices
            user_features_matrix = self.dataset.build_user_features(user_features)
            item_features_matrix = self.dataset.build_item_features(item_features)
            
            self.user_features = user_features_matrix
            self.item_features = item_features_matrix
            
            # Store mappings
            self.user_id_map = self.dataset.mapping()[0]
            self.item_id_map = self.dataset.mapping()[2]
            self.reverse_user_id_map = {v: k for k, v in self.user_id_map.items()}
            self.reverse_item_id_map = {v: k for k, v in self.item_id_map.items()}
            
            return interactions_matrix, weights
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    def train(self, epochs=50, num_threads=2):
        """Train the LightFM model"""
        try:
            data = self.prepare_data()
            if not data:
                return False
                
            interactions_matrix, weights = data
            
            # Initialize model
            self.model = LightFM(
                loss='warp',
                learning_rate=0.05,
                item_alpha=1e-6,
                user_alpha=1e-6,
                no_components=50
            )
            
            # Train model
            self.model.fit(
                interactions_matrix,
                user_features=self.user_features,
                item_features=self.item_features,
                epochs=epochs,
                num_threads=num_threads,
                verbose=True
            )
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, user_id, num_recommendations=10):
        """Generate recommendations for a user"""
        try:
            # Import models here to avoid circular imports
            from .models import Movie, Rating
            
            if not self.model or user_id not in self.user_id_map:
                return []
            
            internal_user_id = self.user_id_map[user_id]
            
            # Get all items
            all_items = list(self.item_id_map.keys())
            internal_item_ids = [self.item_id_map[item_id] for item_id in all_items]
            
            # Get predictions
            scores = self.model.predict(
                internal_user_id,
                internal_item_ids,
                user_features=self.user_features,
                item_features=self.item_features,
                num_threads=2
            )
            
            # Get user's already rated movies to exclude them
            user_rated_movies = set(
                Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True)
            )
            
            # Create recommendations with scores
            recommendations = []
            for i, score in enumerate(scores):
                movie_id = all_items[i]
                if movie_id not in user_rated_movies:
                    try:
                        movie = Movie.objects.get(id=movie_id)
                        recommendations.append({
                            'movie': movie,
                            'score': float(score),
                            'confidence_score': round(float(score), 3)
                        })
                    except Movie.DoesNotExist:
                        continue
            
            # Sort by score and return top N (remove duplicates)
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Remove duplicates based on movie ID
            seen_movies = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec['movie'].id not in seen_movies:
                    seen_movies.add(rec['movie'].id)
                    unique_recommendations.append(rec)
            
            return unique_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []
    
    def explain(self, user_id, movie_id):
        """Generate explanation for why a movie was recommended"""
        try:
            # Import models here to avoid circular imports
            from .models import Movie, UserProfile
            
            if not self.model or user_id not in self.user_id_map or movie_id not in self.item_id_map:
                return {"error": "Model not trained or user/movie not found"}
            
            internal_user_id = self.user_id_map[user_id]
            internal_movie_id = self.item_id_map[movie_id]
            
            # Get user and item embeddings
            user_embedding = self.model.user_embeddings[internal_user_id]
            item_embedding = self.model.item_embeddings[internal_movie_id]
            
            # Get user and item biases
            user_bias = self.model.user_biases[internal_user_id] if hasattr(self.model, 'user_biases') else 0
            item_bias = self.model.item_biases[internal_movie_id] if hasattr(self.model, 'item_biases') else 0
            
            # Calculate feature contributions (simplified approach)
            feature_contributions = user_embedding * item_embedding
            
            # Get movie and user info for explanation
            movie = Movie.objects.get(id=movie_id)
            user = User.objects.get(id=user_id)
            
            # Get user's preferred genres from profile
            try:
                user_profile = user.userprofile
                if user_profile.preferred_genres:
                    user_genres = [genre.strip() for genre in user_profile.preferred_genres.split(',')]
                else:
                    user_genres = []
            except:
                user_genres = []
            
            # Get movie genres
            movie_genres = movie.genres.split('|') if movie.genres else []
            
            # Find matching genres
            matching_genres = list(set(user_genres) & set(movie_genres))
            
            # Create explanation
            explanation = {
                "prediction_score": round(float(np.dot(user_embedding, item_embedding) + user_bias + item_bias), 3),
                "user_bias": round(float(user_bias), 3),
                "item_bias": round(float(item_bias), 3),
                "matching_genres": matching_genres if matching_genres else ["No direct genre matches"],
                "movie_genres": movie_genres,
                "user_preferred_genres": user_genres,
                "top_feature_contributions": [
                    f"Feature {i}: {round(float(contrib), 3)}" 
                    for i, contrib in enumerate(feature_contributions[:5])
                ]
            }
            
            # Add human-readable explanation
            if matching_genres:
                explanation["reason"] = f"Recommended because you like {', '.join(matching_genres)} movies"
            elif movie.tmdb_rating and float(movie.tmdb_rating) > 7.0:
                explanation["reason"] = f"Recommended because it's a highly rated movie ({movie.tmdb_rating}/10)"
            else:
                explanation["reason"] = "Recommended based on your viewing patterns and preferences"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {"error": f"Could not generate explanation: {str(e)}"}
    
    def save_model(self, filepath):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'dataset': self.dataset,
                'user_features': self.user_features,
                'item_features': self.item_features,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'reverse_user_id_map': self.reverse_user_id_map,
                'reverse_item_id_map': self.reverse_item_id_map,
                'feature_names': self.feature_names
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.dataset = model_data['dataset']
            self.user_features = model_data['user_features']
            self.item_features = model_data['item_features']
            self.user_id_map = model_data['user_id_map']
            self.item_id_map = model_data['item_id_map']
            self.reverse_user_id_map = model_data['reverse_user_id_map']
            self.reverse_item_id_map = model_data['reverse_item_id_map']
            self.feature_names = model_data.get('feature_names', [])
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

# Global engine instance
engine = LightFMEngine()