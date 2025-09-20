# # from django.contrib import admin
# # from django.urls import path, include
# # from django.conf import settings
# # from django.conf.urls.static import static

# # urlpatterns = [
# #     path('admin/', admin.site.urls),
# #     path('', include('apps.core.urls')),
# #     path('accounts/', include('apps.accounts.urls')),
# #     path('movies/', include('apps.movies.urls')),
# #     path('api/', include('apps.recommender.urls')),
# # ]

# # if settings.DEBUG:
# #     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# #     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# # # Admin customization
# # admin.site.site_header = "Movie Recommender Admin"
# # admin.site.site_title = "Movie Recommender"
# # admin.site.index_title = "Welcome to Movie Recommender Administration"
# from django.contrib import admin
# from django.urls import path, include
# from django.shortcuts import redirect

# # Redirect root to appropriate page
# def root_redirect(request):
#     if request.user.is_authenticated:
#         if not request.user.survey_completed:
#             return redirect('recommender:welcome')
#         return redirect('recommender:recommendations')
#     return redirect('accounts:login')

# urlpatterns = [
#     path('', root_redirect, name='root'),  # root redirect
#     path('admin/', admin.site.urls),
#     path('accounts/', include(('apps.accounts.urls', 'accounts'), namespace='accounts')),
#     path('recommender/', include(('apps.recommender.urls', 'recommender'), namespace='recommender')), 
#     path("accounts/", include("django.contrib.auth.urls")), # only once
# ]
from django.contrib import admin
from django.urls import path, include
from apps.core import views as core_views
from apps.movies import views as movie_views
from apps.accounts import views as account_views


urlpatterns = [
    path('', account_views.home, name="home"),
    path("admin/", admin.site.urls),
    path("", core_views.home, name="home"),
    path("accounts/", include("apps.accounts.urls")),
    path("recommender/", include("apps.recommender.urls")),
    path("accounts/", include("django.contrib.auth.urls")), 
    path('', movie_views.home, name='home'), 
    
]