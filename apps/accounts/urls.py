# from django.urls import path
# from . import views

# app_name = "accounts"

# urlpatterns = [
#     path("signup/", views.signup, name="signup"),
#     path("register/", views.register, name="register"),
#     path("login/", views.login_view, name="login"),
#     path("logout/", views.logout_view, name="logout"),
#     path("survey/", views.survey, name="survey"),
#     path("profile/", views.profile_view, name="profile"),
#     path("post-login/", views.post_login_redirect, name="post_login_redirect"),
# ]
from django.urls import path
from . import views

app_name = "accounts"

urlpatterns = [
    path("signup/", views.signup, name="signup"),
    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("survey/", views.survey, name="survey"),
    path("survey/api/", views.survey_api, name="survey_api"),   # ✅ new API endpoint
    path("profile/", views.profile_view, name="profile"),
    path("post-login/", views.post_login_redirect, name="post_login_redirect"),
]