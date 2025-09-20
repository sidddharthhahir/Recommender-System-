# # import os
# # import logging
# # from django.core.management.base import BaseCommand
# # from django.conf import settings
# # from django.contrib.auth import get_user_model
# # from apps.accounts.models import Profile
# # from apps.recommender.lightfm_engine import lightfm_engine
# # from apps.recommender.models import ModelVersion

# # logger = logging.getLogger("recommender")
# # User = get_user_model()


# # class Command(BaseCommand):
# #     help = "Train the LightFM recommendation model and initialize explainers."

# #     def add_arguments(self, parser):
# #         parser.add_argument(
# #             '--epochs',
# #             type=int,
# #             default=20,
# #             help='Number of training epochs (default: 20)'
# #         )
# #         parser.add_argument(
# #             '--components',
# #             type=int,
# #             default=50,
# #             help='Number of latent components (default: 50)'
# #         )

# #     def handle(self, *args, **options):
# #         try:
# #             epochs = options['epochs']
# #             components = options['components']
            
# #             self.stdout.write("Starting LightFM model training...")
# #             logger.info("Starting LightFM model training...")

# #             # Train the model and get training stats
# #             metrics, training_stats = lightfm_engine.train_with_stats(
# #                 epochs=epochs, 
# #                 num_components=components
# #             )

# #             # Save the model file
# #             model_dir = os.path.join(settings.BASE_DIR, "models")
# #             os.makedirs(model_dir, exist_ok=True)
# #             filepath = os.path.join(model_dir, "lightfm_model.pkl")
# #             lightfm_engine.save_model(filepath)

# #             self.stdout.write(self.style.SUCCESS(f"✅ Model saved to {filepath}"))
# #             logger.info("Initializing explainers...")

# #             # Safely prepare user features for explainers
# #             user_features = []
# #             for user in User.objects.all():
# #                 if hasattr(user, "profile"):
# #                     profile = user.profile
# #                     features = {
# #                         "birth_date": profile.birth_date.isoformat()
# #                         if profile.birth_date else None,
# #                         "favorite_genres": profile.favorite_genres or [],
# #                         "preferred_languages": profile.preferred_languages or [],
# #                         "openness": getattr(profile, "openness_to_new", None),
# #                     }
# #                 else:
# #                     features = {
# #                         "birth_date": None,
# #                         "favorite_genres": [],
# #                         "preferred_languages": [],
# #                         "openness": None,
# #                     }
# #                 user_features.append(features)

# #             self.stdout.write("Initializing recommendation explainers...")
# #             logger.info("Initializing recommendation explainers...")

# #             # Initialize explainers (placeholder for now)
# #             self._init_explainers(lightfm_engine, user_features)

# #             # Save a version record in DB with all required fields
# #             mv = ModelVersion.objects.create(
# #                 version_name=f"v{ModelVersion.objects.count() + 1}",
# #                 algorithm="lightfm",
# #                 training_data_size=training_stats.get('training_data_size', 0),
# #                 num_users=training_stats.get('num_users', 0),
# #                 num_items=training_stats.get('num_items', 0),
# #                 num_interactions=training_stats.get('num_interactions', 0),
# #                 num_epochs=epochs,
# #                 num_components=components,
# #                 learning_rate=0.05,  # default from lightfm_engine
# #                 loss_function='warp',  # default from lightfm_engine
# #                 metrics=metrics,
# #             )
            
# #             self.stdout.write(
# #                 self.style.SUCCESS(
# #                     f"✅ Model training logged in DB as version {mv.version_name}"
# #                 )
# #             )
# #             self.stdout.write(
# #                 self.style.SUCCESS(
# #                     f"📊 Training stats: {training_stats['num_users']} users, "
# #                     f"{training_stats['num_items']} items, "
# #                     f"{training_stats['num_interactions']} interactions"
# #                 )
# #             )

# #         except Exception as e:
# #             logger.error(f"Model training failed: {e}", exc_info=True)
# #             self.stdout.write(self.style.ERROR(f"❌ Model training failed: {e}"))

# #     def _init_explainers(self, engine, user_features):
# #         """
# #         Initialize SHAP, LIME, Anchors explainers.
# #         Placeholder implementation - you can expand this for actual XAI setup.
# #         """
# #         logger.info("Explainability modules initialized (placeholder).")
# #         self.stdout.write("✅ Explainers initialized successfully")
# import os
# import logging
# import time
# from django.core.management.base import BaseCommand
# from django.conf import settings
# from django.contrib.auth import get_user_model
# from apps.accounts.models import Profile
# from apps.recommender.engine import engine, explanation_engine
# from apps.recommender.models import ModelVersion
# logger = logging.getLogger("recommender")
# User = get_user_model()


# class Command(BaseCommand):
#     help = "Train the LightFM recommendation model and initialize explainers."

#     def add_arguments(self, parser):
#         parser.add_argument(
#             '--epochs',
#             type=int,
#             default=20,
#             help='Number of training epochs (default: 20)'
#         )
#         parser.add_argument(
#             '--components',
#             type=int,
#             default=50,
#             help='Number of latent components (default: 50)'
#         )

#     def handle(self, *args, **options):
#         try:
#             epochs = options['epochs']
#             components = options['components']
            
#             self.stdout.write("🚀 Starting LightFM model training...")
#             logger.info("Starting LightFM model training...")

#             # Record training start time
#             start_time = time.time()

#             # Train the model and get training stats
#             metrics, training_stats = lightfm_engine.train_with_stats(
#                 epochs=epochs, 
#                 num_components=components
#             )

#             # Calculate training duration
#             training_duration = (time.time() - start_time) / 60  # Convert to minutes

#             # Save the model file
#             model_dir = os.path.join(settings.BASE_DIR, "models")
#             os.makedirs(model_dir, exist_ok=True)
#             filepath = os.path.join(model_dir, "lightfm_model.pkl")
#             lightfm_engine.save_model(filepath)

#             self.stdout.write(self.style.SUCCESS(f"✅ Model saved to {filepath}"))
            
#             # Initialize explainers
#             self.stdout.write("🧠 Initializing SHAP & LIME explainers...")
#             logger.info("Initializing explainers...")

#             explainer_success = explanation_engine.init_explainers()
            
#             if explainer_success:
#                 self.stdout.write(self.style.SUCCESS("✅ SHAP & LIME explainers initialized successfully"))
#             else:
#                 self.stdout.write(self.style.WARNING("⚠️ Explainer initialization failed"))

#             # Extract performance metrics from the metrics dict
#             precision_at_10 = metrics.get('test_precision_at_10')
#             recall_at_10 = metrics.get('test_recall_at_10')
#             auc_score = metrics.get('test_auc')

#             # Save a version record in DB with all required fields
#             mv = ModelVersion.objects.create(
#                 version_name=f"v{ModelVersion.objects.count() + 1}",
#                 algorithm="lightfm",
#                 training_data_size=training_stats.get('training_data_size', 0),
#                 num_users=training_stats.get('num_users', 0),
#                 num_items=training_stats.get('num_items', 0),
#                 num_interactions=training_stats.get('num_interactions', 0),
#                 num_epochs=epochs,
#                 num_components=components,
#                 learning_rate=0.05,  # default from lightfm_engine
#                 loss_function='warp',  # default from lightfm_engine
#                 training_duration_minutes=training_duration,
#                 hyperparameters={
#                     'epochs': epochs,
#                     'components': components,
#                     'learning_rate': 0.05,
#                     'loss': 'warp'
#                 },
#                 precision_at_10=precision_at_10,
#                 recall_at_10=recall_at_10,
#                 auc_score=auc_score,
#                 metrics=metrics,
#                 model_file_path=filepath,
#                 is_active=True  # Mark as active model
#             )
            
#             # Deactivate previous models
#             ModelVersion.objects.filter(is_active=True).exclude(id=mv.id).update(is_active=False)
            
#             self.stdout.write(
#                 self.style.SUCCESS(
#                     f"✅ Model training logged in DB as version {mv.version_name}"
#                 )
#             )
#             self.stdout.write(
#                 self.style.SUCCESS(
#                     f"📊 Training stats: {training_stats['num_users']} users, "
#                     f"{training_stats['num_items']} items, "
#                     f"{training_stats['num_interactions']} interactions"
#                 )
#             )
#             self.stdout.write(
#                 self.style.SUCCESS(
#                     f"⏱️ Training completed in {training_duration:.2f} minutes"
#                 )
#             )

#         except Exception as e:
#             logger.error(f"Model training failed: {e}", exc_info=True)
#             self.stdout.write(self.style.ERROR(f"❌ Model training failed: {e}"))
from django.core.management.base import BaseCommand
import logging
import time

from apps.recommender.engine import engine

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Train the LightFM model and save it"

    def handle(self, *args, **options):
        self.stdout.write("🚀 Starting LightFM model training...")

        start_time = time.time()
        try:
            # Train the model using the engine's train method
            success = engine.train()
            if not success:
                self.stderr.write("❌ Model training failed.")
                return

            # Save the trained model to disk
            model_path = "apps/recommender/models/lightfm_model.pkl"
            saved = engine.save_model(model_path)
            if saved:
                self.stdout.write(f"✅ Model saved successfully to {model_path}")
            else:
                self.stderr.write("❌ Failed to save the model.")

            elapsed = time.time() - start_time
            self.stdout.write(f"⏱ Training completed in {elapsed:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            self.stderr.write(f"❌ Exception during training: {e}")