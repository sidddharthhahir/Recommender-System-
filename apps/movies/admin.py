from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Avg, Count
from .models import Genre, Movie, Rating, Watchlist, MovieInteraction

@admin.register(Genre)
class GenreAdmin(admin.ModelAdmin):
    list_display = ('name', 'tmdb_id', 'movie_count')
    search_fields = ('name',)
    ordering = ('name',)
    
    def movie_count(self, obj):
        return obj.movie_set.count()
    movie_count.short_description = 'Movies'

@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ('title', 'release_year', 'tmdb_rating', 'user_rating', 'total_ratings', 'popularity_score', 'poster_preview')
    list_filter = ('genres', 'release_date', 'original_language', 'adult', 'is_active')
    search_fields = ('title', 'original_title', 'tmdb_id')
    readonly_fields = ('tmdb_id', 'avg_user_rating', 'total_user_ratings', 'created_at', 'last_updated', 'poster_preview', 'backdrop_preview')
    filter_horizontal = ('genres',)
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'original_title', 'tmdb_id', 'overview')
        }),
        ('Release & Runtime', {
            'fields': ('release_date', 'runtime', 'original_language', 'adult')
        }),
        ('Ratings & Popularity', {
            'fields': ('tmdb_rating', 'tmdb_vote_count', 'popularity', 'avg_user_rating', 'total_user_ratings')
        }),
        ('Media', {
            'fields': ('poster_path', 'poster_preview', 'backdrop_path', 'backdrop_preview'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('genres', 'cast', 'crew', 'keywords', 'production_companies'),
            'classes': ('collapse',)
        }),
        ('System', {
            'fields': ('is_active', 'recommendation_score', 'created_at', 'last_updated'),
            'classes': ('collapse',)
        }),
    )
    
    def release_year(self, obj):
        return obj.release_date.year if obj.release_date else "Unknown"
    release_year.short_description = 'Year'
    
    def user_rating(self, obj):
        return f"{obj.avg_user_rating:.1f}" if obj.avg_user_rating else "N/A"
    user_rating.short_description = 'User Rating'
    
    def total_ratings(self, obj):
        return obj.total_user_ratings
    total_ratings.short_description = 'User Ratings'
    
    def popularity_score(self, obj):
        return f"{obj.popularity:.1f}"
    popularity_score.short_description = 'Popularity'
    
    def poster_preview(self, obj):
        if obj.poster_url:
            return format_html('<img src="{}" width="50" height="75" />', obj.poster_url)
        return "No poster"
    poster_preview.short_description = 'Poster'
    
    def backdrop_preview(self, obj):
        if obj.backdrop_url:
            return format_html('<img src="{}" width="100" height="56" />', obj.backdrop_url)
        return "No backdrop"
    backdrop_preview.short_description = 'Backdrop'

@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie', 'rating', 'would_recommend', 'created_at')
    list_filter = ('rating', 'would_recommend', 'created_at')
    search_fields = ('user__username', 'movie__title')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Rating', {
            'fields': ('user', 'movie', 'rating', 'would_recommend')
        }),
        ('Review', {
            'fields': ('review',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie', 'added_at')
    list_filter = ('added_at',)
    search_fields = ('user__username', 'movie__title')
    readonly_fields = ('added_at',)

@admin.register(MovieInteraction)
class MovieInteractionAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie', 'interaction_type', 'source_page', 'created_at')
    list_filter = ('interaction_type', 'source_page', 'created_at')
    search_fields = ('user__username', 'movie__title')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Interaction', {
            'fields': ('user', 'movie', 'interaction_type')
        }),
        ('Context', {
            'fields': ('source_page', 'session_id')
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        }),
    )