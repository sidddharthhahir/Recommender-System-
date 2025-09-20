# from django.contrib import admin
# from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
# from django.contrib.auth.models import User
# from django.utils.html import format_html
# from .models import Profile, UserPreference


# # Inline admin for Profile (shows up in User admin)
# class ProfileInline(admin.StackedInline):
#     model = Profile
#     can_delete = False
#     verbose_name_plural = 'Profile'
#     fields = (
#         ('date_of_birth', 'survey_completed'),
#         ('favorite_genres', 'favorite_actors', 'preferred_languages'),
#         ('openness_to_new', 'preferred_decade', 'min_rating_threshold'),
#         ('total_recommendations_received', 'total_ratings_given', 'avg_rating_given')
#     )
#     readonly_fields = ('total_recommendations_received', 'total_ratings_given', 'avg_rating_given')


# # Extend the default UserAdmin to include Profile
# class UserAdmin(BaseUserAdmin):
#     inlines = (ProfileInline,)
#     list_display = ('username', 'email', 'survey_status', 'total_ratings', 'avg_rating', 'date_joined')
#     list_filter = ('is_staff', 'is_active', 'date_joined')
    
#     def survey_status(self, obj):
#         try:
#             if obj.profile.survey_completed:
#                 return format_html('<span style="color: green;">✓ Completed</span>')
#             return format_html('<span style="color: red;">✗ Pending</span>')
#         except Profile.DoesNotExist:
#             return format_html('<span style="color: gray;">No Profile</span>')
#     survey_status.short_description = 'Survey'
    
#     def total_ratings(self, obj):
#         try:
#             return obj.profile.total_ratings_given
#         except Profile.DoesNotExist:
#             return 0
#     total_ratings.short_description = 'Ratings Given'
    
#     def avg_rating(self, obj):
#         try:
#             return f"{obj.profile.avg_rating_given:.1f}" if obj.profile.avg_rating_given else "N/A"
#         except Profile.DoesNotExist:
#             return "N/A"
#     avg_rating.short_description = 'Avg Rating'


# # Re-register UserAdmin
# admin.site.unregister(User)
# admin.site.register(User, UserAdmin)


# @admin.register(Profile)
# class ProfileAdmin(admin.ModelAdmin):
#     list_display = ('user', 'survey_completed', 'openness_to_new', 
#                     'min_rating_threshold', 'total_recommendations_received', 'created_at')
#     list_filter = ('survey_completed', 'preferred_decade', 'created_at')
#     search_fields = ('user__username', 'user__email', 'user__first_name', 'user__last_name')
#     readonly_fields = ('total_recommendations_received', 'total_ratings_given', 'avg_rating_given', 'created_at', 'updated_at')
    
#     fieldsets = (
#         ('User', {
#             'fields': ('user',)
#         }),
#         ('Profile Information', {
#             'fields': ('date_of_birth', 'favorite_genres', 'favorite_actors', 'preferred_languages')
#         }),
#         ('Recommendation Settings', {
#             'fields': ('openness_to_new', 'preferred_decade', 'min_rating_threshold')
#         }),
#         ('Survey Status', {
#             'fields': ('survey_completed', 'survey_completed_at')
#         }),
#         ('Statistics', {
#             'fields': ('total_recommendations_received', 'total_ratings_given', 'avg_rating_given'),
#             'classes': ('collapse',)
#         }),
#         ('Timestamps', {
#             'fields': ('created_at', 'updated_at'),
#             'classes': ('collapse',)
#         }),
#     )


# @admin.register(UserPreference)
# class UserPreferenceAdmin(admin.ModelAdmin):
#     list_display = ('profile', 'favorite_movie', 'movie_frequency', 'discovery_method', 'created_at')
#     list_filter = ('movie_frequency', 'discovery_method', 'created_at')
#     search_fields = ('profile__user__username', 'favorite_movie')
#     readonly_fields = ('created_at', 'updated_at')
    
#     fieldsets = (
#         ('Profile', {
#             'fields': ('profile',)
#         }),
#         ('Survey Responses', {
#             'fields': ('favorite_movie', 'least_favorite_genre', 'movie_frequency', 'discovery_method')
#         }),
#         ('Learned Preferences', {
#             'fields': ('genre_weights', 'actor_weights', 'director_weights'),
#             'classes': ('collapse',)
#         }),
#         ('Timestamps', {
#             'fields': ('created_at', 'updated_at'),
#             'classes': ('collapse',)
#         }),
#     )
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import Profile

class ProfileInline(admin.StackedInline):
    model = Profile
    can_delete = False
    verbose_name_plural = 'Profile'
    extra = 0
    readonly_fields = ['created_at', 'updated_at']  # only fields that actually exist

class ProfileAdmin(admin.ModelAdmin):
    model = Profile
    list_display = ['user', 'birth_date', 'openness_to_new', 'created_at']
    search_fields = ['user__username', 'user__email']
    list_filter = ['openness_to_new', 'created_at']
    readonly_fields = ['created_at', 'updated_at']

# Re-register UserAdmin with Profile inlines
class UserAdmin(BaseUserAdmin):
    inlines = (ProfileInline, )

# Unregister the default User admin and re-register with our custom one
admin.site.unregister(User)
admin.site.register(User, UserAdmin)
admin.site.register(Profile, ProfileAdmin)