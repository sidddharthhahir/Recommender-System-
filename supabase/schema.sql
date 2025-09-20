-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (extends Django's auth_user)
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL,
    date_of_birth DATE,
    favorite_genres JSONB DEFAULT '[]',
    favorite_actors JSONB DEFAULT '[]',
    preferred_languages JSONB DEFAULT '[]',
    openness_to_new INTEGER DEFAULT 5,
    preferred_decade VARCHAR(10),
    min_rating_threshold FLOAT DEFAULT 6.0,
    survey_completed BOOLEAN DEFAULT FALSE,
    survey_completed_at TIMESTAMP,
    total_recommendations_received INTEGER DEFAULT 0,
    total_ratings_given INTEGER DEFAULT 0,
    avg_rating_given FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Genres table
CREATE TABLE IF NOT EXISTS genres (
    id SERIAL PRIMARY KEY,
    tmdb_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Movies table
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    tmdb_id INTEGER UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    original_title VARCHAR(500),
    overview TEXT,
    release_date DATE,
    runtime INTEGER,
    tmdb_rating FLOAT DEFAULT 0.0,
    tmdb_vote_count INTEGER DEFAULT 0,
    popularity FLOAT DEFAULT 0.0,
    poster_path VARCHAR(200),
    backdrop_path VARCHAR(200),
    original_language VARCHAR(10),
    adult BOOLEAN DEFAULT FALSE,
    cast JSONB DEFAULT '[]',
    crew JSONB DEFAULT '[]',
    keywords JSONB DEFAULT '[]',
    production_companies JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    avg_user_rating FLOAT DEFAULT 0.0,
    total_user_ratings INTEGER DEFAULT 0,
    recommendation_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Movie-Genre relationship
CREATE TABLE IF NOT EXISTS movie_genres (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    genre_id INTEGER REFERENCES genres(id) ON DELETE CASCADE,
    UNIQUE(movie_id, genre_id)
);

-- Ratings table
CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review TEXT,
    would_recommend BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, movie_id)
);

-- Watchlist table
CREATE TABLE IF NOT EXISTS watchlist (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    added_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, movie_id)
);

-- Movie interactions table
CREATE TABLE IF NOT EXISTS movie_interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    interaction_type VARCHAR(20) NOT NULL,
    source_page VARCHAR(100),
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Recommendation logs table
CREATE TABLE IF NOT EXISTS recommendation_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    request_type VARCHAR(50) DEFAULT 'homepage',
    algorithm_version VARCHAR(20) DEFAULT 'lightfm_v1',
    total_movies INTEGER DEFAULT 0,
    avg_confidence FLOAT DEFAULT 0.0,
    processing_time_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Recommendation items table
CREATE TABLE IF NOT EXISTS recommendation_items (
    id SERIAL PRIMARY KEY,
    log_id INTEGER REFERENCES recommendation_logs(id) ON DELETE CASCADE,
    movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    confidence_score FLOAT NOT NULL,
    clicked BOOLEAN DEFAULT FALSE,
    clicked_at TIMESTAMP,
    rated BOOLEAN DEFAULT FALSE,
    rated_at TIMESTAMP,
    UNIQUE(log_id, movie_id)
);

-- Explanation logs table
CREATE TABLE IF NOT EXISTS explanation_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
    recommendation_log_id INTEGER REFERENCES recommendation_logs(id) ON DELETE SET NULL,
    explanation_type VARCHAR(20) NOT NULL,
    explanation_data JSONB NOT NULL,
    viewed BOOLEAN DEFAULT TRUE,
    clicked_details BOOLEAN DEFAULT FALSE,
    helpful_rating INTEGER CHECK (helpful_rating >= 1 AND helpful_rating <= 5),
    generation_time_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version_name VARCHAR(50) UNIQUE NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    training_data_size INTEGER NOT NULL,
    training_duration_minutes FLOAT NOT NULL,
    hyperparameters JSONB NOT NULL,
    precision_at_10 FLOAT,
    recall_at_10 FLOAT,
    auc_score FLOAT,
    is_active BOOLEAN DEFAULT FALSE,
    model_file_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW()
);

-- User embeddings table
CREATE TABLE IF NOT EXISTS user_embeddings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL,
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE CASCADE,
    embedding_vector JSONB NOT NULL,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Movie embeddings table
CREATE TABLE IF NOT EXISTS movie_embeddings (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER UNIQUE REFERENCES movies(id) ON DELETE CASCADE,
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE CASCADE,
    embedding_vector JSONB NOT NULL,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Survey responses table
CREATE TABLE IF NOT EXISTS survey_responses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL,
    favorite_genres JSONB NOT NULL,
    favorite_actor VARCHAR(200),
    last_loved_movie VARCHAR(200),
    rating_style VARCHAR(50),
    openness_to_new INTEGER NOT NULL,
    seed_ratings_generated BOOLEAN DEFAULT FALSE,
    seed_ratings_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies(tmdb_id);
CREATE INDEX IF NOT EXISTS idx_movies_popularity ON movies(popularity DESC);
CREATE INDEX IF NOT EXISTS idx_movies_rating ON movies(tmdb_rating DESC);
CREATE INDEX IF NOT EXISTS idx_movies_release_date ON movies(release_date DESC);
CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_ratings_created_at ON ratings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_logs_user_id ON recommendation_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_logs_created_at ON recommendation_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_explanation_logs_user_id ON explanation_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_explanation_logs_movie_id ON explanation_logs(movie_id);

-- Enable Row Level Security (RLS)
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE movie_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendation_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE explanation_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE survey_responses ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (users can only access their own data)
CREATE POLICY "Users can view own profile" ON user_profiles FOR SELECT USING (auth.uid()::text = user_id::text);
CREATE POLICY "Users can update own profile" ON user_profiles FOR UPDATE USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can view own ratings" ON ratings FOR SELECT USING (auth.uid()::text = user_id::text);
CREATE POLICY "Users can insert own ratings" ON ratings FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);
CREATE POLICY "Users can update own ratings" ON ratings FOR UPDATE USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can view own watchlist" ON watchlist FOR SELECT USING (auth.uid()::text = user_id::text);
CREATE POLICY "Users can manage own watchlist" ON watchlist FOR ALL USING (auth.uid()::text = user_id::text);

-- Functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for auto-updating timestamps
CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_movies_updated_at BEFORE UPDATE ON movies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ratings_updated_at BEFORE UPDATE ON ratings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();