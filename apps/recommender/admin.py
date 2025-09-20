from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Avg, Count
from .models import (
    RecommendationLog, RecommendationItem, ExplanationLog, 
    ModelVersion, UserEmbedding, MovieEmbedding, SurveyResponse
)

@admin.register(RecommendationLog)
class RecommendationLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'request_type', 'total_movies', 'avg_confidence', 'processing_time', 'created_at')
    list_filter = ('request_type', 'algorithm_version', 'created_at')
    search_fields = ('user__username',)
    readonly_fields = ('created_at',)
    
    def processing_time(self, obj):
        return f"{obj.processing_time_ms}ms"
    processing_time.short_description = 'Processing Time'

class RecommendationItemInline(admin.TabularInline):
    model = RecommendationItem
    extra = 0
    readonly_fields = ('clicked_at', 'rated_at')
    fields = ('movie', 'position', 'confidence_score', 'clicked', 'clicked_at', 'rated', 'rated_at')

@admin.register(RecommendationItem)
class RecommendationItemAdmin(admin.ModelAdmin):
    list_display = ('movie', 'user', 'position', 'confidence_score', 'interaction_status', 'log_date')
    list_filter = ('clicked', 'rated', 'log__created_at')
    search_fields = ('movie__title', 'log__user__username')
    
    def user(self, obj):
        return obj.log.user.username
    user.short_description = 'User'
    
    def interaction_status(self, obj):
        status = []
        if obj.clicked:
            status.append('Clicked')
        if obj.rated:
            status.append('Rated')
        return ', '.join(status) if status else 'No interaction'
    interaction_status.short_description = 'Interaction'
    
    def log_date(self, obj):
        return obj.log.created_at
    log_date.short_description = 'Recommended At'

@admin.register(ExplanationLog)
class ExplanationLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie', 'explanation_type', 'viewed', 'clicked_details', 'helpful_rating', 'created_at')
    list_filter = ('explanation_type', 'viewed', 'clicked_details', 'helpful_rating', 'created_at')
    search_fields = ('user__username', 'movie__title')
    readonly_fields = ('created_at', 'generation_time_ms')
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'movie', 'recommendation_log', 'explanation_type')
        }),
        ('Explanation Data', {
            'fields': ('explanation_data',),
            'classes': ('collapse',)
        }),
        ('User Interaction', {
            'fields': ('viewed', 'clicked_details', 'helpful_rating')
        }),
        ('Metadata', {
            'fields': ('generation_time_ms', 'created_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('version_name', 'algorithm', 'status', 'performance_summary', 'training_duration', 'created_at')
    list_filter = ('algorithm', 'is_active', 'created_at')
    search_fields = ('version_name', 'algorithm')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Model Info', {
            'fields': ('version_name', 'algorithm', 'is_active', 'model_file_path')
        }),
        ('Training Details', {
            'fields': ('training_data_size', 'training_duration_minutes', 'hyperparameters'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': ('precision_at_10', 'recall_at_10', 'auc_score')
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def status(self, obj):
        if obj.is_active:
            return format_html('<span style="color: green; font-weight: bold;">ACTIVE</span>')
        return format_html('<span style="color: gray;">Inactive</span>')
    status.short_description = 'Status'
    
    def performance_summary(self, obj):
        if obj.precision_at_10 and obj.recall_at_10:
            return f"P@10: {obj.precision_at_10:.3f}, R@10: {obj.recall_at_10:.3f}"
        return "No metrics"
    performance_summary.short_description = 'Performance'
    
    def training_duration(self, obj):
        return f"{obj.training_duration_minutes:.1f} min"
    training_duration.short_description = 'Training Time'

@admin.register(UserEmbedding)
class UserEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('user', 'model_version', 'embedding_dimension', 'last_updated')
    list_filter = ('model_version', 'last_updated')
    search_fields = ('user__username',)
    readonly_fields = ('last_updated',)
    
    def embedding_dimension(self, obj):
        return len(obj.embedding_vector) if obj.embedding_vector else 0
    embedding_dimension.short_description = 'Dimensions'

@admin.register(MovieEmbedding)
class MovieEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('movie', 'model_version', 'embedding_dimension', 'last_updated')
    list_filter = ('model_version', 'last_updated')
    search_fields = ('movie__title',)
    readonly_fields = ('last_updated',)
    
    def embedding_dimension(self, obj):
        return len(obj.embedding_vector) if obj.embedding_vector else 0
    embedding_dimension.short_description = 'Dimensions'

@admin.register(SurveyResponse)
class SurveyResponseAdmin(admin.ModelAdmin):
    list_display = ('user', 'favorite_actor', 'openness_to_new', 'seed_status', 'created_at')
    list_filter = ('seed_ratings_generated', 'openness_to_new', 'created_at')
    search_fields = ('user__username', 'favorite_actor', 'last_loved_movie')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Survey Responses', {
            'fields': ('favorite_genres', 'favorite_actor', 'last_loved_movie', 'rating_style', 'openness_to_new')
        }),
        ('Seed Ratings', {
            'fields': ('seed_ratings_generated', 'seed_ratings_count')
        }),
        ('Timestamp', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def seed_status(self, obj):
        if obj.seed_ratings_generated:
            return format_html('<span style="color: green;">✓ Generated ({})</span>', obj.seed_ratings_count)
        return format_html('<span style="color: red;">✗ Pending</span>')
    seed_status.short_description = 'Seed Ratings'