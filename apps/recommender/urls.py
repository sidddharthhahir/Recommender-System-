# # # from django.urls import path
# # # from . import views

# # # app_name = 'recommender'

# # # urlpatterns = [
# # #     # API endpoints
# # #     path('recommendations/', views.get_recommendations, name='get_recommendations'),
# # #     path('explanations/<int:movie_id>/', views.get_explanation, name='get_explanation'),
# # #     path('ratings/', views.submit_rating, name='submit_rating'),
# # #     path('survey/', views.submit_survey, name='submit_survey'),
# # #     path('interactions/', views.log_interaction, name='log_interaction'),
    
# # #     # Web pages
# # #     path('welcome/', views.welcome, name='welcome'),
# # #     path('recommendations/', views.recommendations, name='recommendations'),
# # # ]
# # from django.urls import path
# # from . import views

# # app_name = 'recommender'

# # urlpatterns = [
# #     path('recommendations/', views.get_recommendations, name='recommendations'),
# #     path('explain/', views.explain_recommendation, name='explain'),
# #     path('explanations/history/', views.explanation_history, name='explanation_history'),
# # ]
# from django.urls import path
# from . import views

# app_name = 'recommender'

# urlpatterns = [
#     path('', views.home, name='home'),
#     path('recommendations/', views.recommendations, name='recommendations'),
#     path('explain/<int:movie_id>/', views.explain_recommendation, name='explain'),
#     path('movie/<int:movie_id>/', views.movie_detail, name='movie_detail'),
#     path('movies/', views.movie_list, name='movie_list'),
#     path('profile/', views.user_profile, name='user_profile'),
#     path('history/', views.explanation_history, name='explanation_history'),
#     path('rate-explanation/', views.rate_explanation, name='rate_explanation'),
# ]
from django.urls import path
from . import views

app_name = 'recommender'

urlpatterns = [
    path('', views.home, name='home'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('explain/<int:movie_id>/', views.explain_recommendation, name='explain'),
    path('movie/<int:movie_id>/', views.movie_detail, name='movie_detail'),
    path('movies/', views.movie_list, name='movie_list'),
    path('profile/', views.user_profile, name='user_profile'),
    path('history/', views.explanation_history, name='explanation_history'),
    path('rate-explanation/', views.rate_explanation, name='rate_explanation'),
    path('delete-explanation/<int:explanation_id>/', views.delete_explanation, name='delete_explanation'),
    path('clear-history/', views.clear_explanation_history, name='clear_history'),
]