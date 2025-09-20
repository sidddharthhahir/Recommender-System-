# 🎬 Transparent Movie Recommender

A Netflix-style Django application with LightFM recommendations and multi-layer explainability (SHAP, LIME, Anchors, Counterfactuals).

## 🚀 Features

- **Netflix-style UI** with dark theme and responsive grid
- **Supabase Integration** for auth, database, and real-time updates
- **TMDB API** for rich movie metadata
- **Cold-start Survey** for new users
- **LightFM Hybrid Recommender** (collaborative + content-based)
- **Multi-layer Explanations**:
  - SHAP (global feature importance)
  - LIME (local explanations)
  - Anchors (rule-based explanations)
  - Counterfactual explanations
- **Admin Panel** for managing movies, users, and recommendations
- **Real-time Updates** when users rate movies

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend assets)
- Redis (for caching)
- Supabase account
- TMDB API key

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd transparent-movie-recommender