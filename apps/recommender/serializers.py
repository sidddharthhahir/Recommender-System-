from rest_framework import serializers
from apps.movies.models import Movie, Rating
from .models import RecommendationLog, ExplanationLog

class MovieSerializer(serializers.ModelSerializer):
    genres = serializers.SerializerMethodField()
    
    class Meta:
        model = Movie
        fields = [
            'id', 'title', 'overview', 'release_date', 'runtime',
            'tmdb_rating', 'popularity', 'poster_url', 'backdrop_url',
            'genres'
        ]
    
    def get_genres(self, obj):
        return obj.get_genre_names()

class RecommendationSerializer(serializers.Serializer):
    movie = MovieSerializer()
    score = serializers.FloatField()
    rank = serializers.IntegerField()
    explanation_available = serializers.BooleanField(default=True)

class ExplanationSerializer(serializers.Serializer):
    user_id = serializers.IntegerField()
    movie_id = serializers.IntegerField()
    movie_title = serializers.CharField()
    prediction_score = serializers.FloatField()
    shap = serializers.DictField(required=False)
    lime = serializers.DictField(required=False)
    anchor = serializers.DictField(required=False)
    counterfactual = serializers.DictField(required=False)