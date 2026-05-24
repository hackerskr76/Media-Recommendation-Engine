import streamlit as st
import pandas as pd
import numpy as np
import os
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Set up page configurations
st.set_page_config(page_title="Media Matcher Workspace", layout="wide", page_icon="🎬")

# --- DATA PREPROCESSING WITH CACHING ---
@st.cache_data
def load_and_clean_data():
    # 1. Movie Data Processing
    df_movies = pd.read_csv('tmdb_5000_movies.csv')
    df_credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = df_movies.merge(df_credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

    def parse_json(obj): return [i['name'] for i in ast.literal_eval(obj)]
    def parse_cast(obj): return [i['name'] for idx, i in enumerate(ast.literal_eval(obj)) if idx < 3]
    def parse_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director': return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(parse_json)
    movies['keywords'] = movies['keywords'].apply(parse_json)
    movies['cast'] = movies['cast'].apply(parse_cast)
    movies['crew'] = movies['crew'].apply(parse_director)

    def collapse(L): return [i.replace(" ", "") for i in L]
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(collapse)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movie_db = movies[['movie_id', 'title', 'tags']].copy()
    movie_db['tags'] = movie_db['tags'].apply(lambda x: " ".join(x).lower())

    stemmer = PorterStemmer()
    movie_db['tags'] = movie_db['tags'].apply(lambda text: " ".join([stemmer.stem(w) for w in text.split()]))

    # 2. Song Data Processing
    songs_10s = pd.read_csv('top10s.csv', encoding='latin-1')
    songs_50s = pd.read_csv('top50contry.csv', encoding='latin-1')
    songs_10s = songs_10s.drop(['year', 'artist', 'Unnamed: 0'], axis=1, errors='ignore')
    songs_50s = songs_50s.drop(['year', 'artist', 'Unnamed: 0', 'country', 'added'], axis=1, errors='ignore')
    song_db = pd.concat([songs_10s, songs_50s], ignore_index=True).drop_duplicates(subset=['title']).reset_index(drop=True)

    return movie_db, song_db

# Try to load data assets safely
try:
    movie_db, song_db = load_and_clean_data()
except Exception as e:
    st.error("Error loading CSV files. Please check your data directory.")
    st.stop()

# Build vector space once for performance caching
vectorizer = CountVectorizer(max_features=3000, stop_words='english')
movie_sparse_matrix = vectorizer.fit_transform(movie_db['tags'])

# --- SIDEBAR NAVIGATOR ---
st.sidebar.title("Navigation Dashboard")
app_mode = st.sidebar.selectbox("Choose a Recommender Page:", ["🎬 Movie Matcher", "🎵 Rhythm Recommender"])

# --- PAGE 1: MOVIE RECOMMENDER ---
if app_mode == "🎬 Movie Matcher":
    st.title("🎬 Movie Matcher")
    st.markdown("Select a movie you love, and the engine will match plot keywords, genres, cast, and directors.")

    movie_selected = st.selectbox("Type or select a movie:", movie_db['title'].values)

    if st.button("Generate Movie Recommendations"):
        movie_idx = movie_db[movie_db['title'] == movie_selected].index[0]
        target_vector = movie_sparse_matrix[movie_idx]
        similarity_scores = cosine_similarity(target_vector, movie_sparse_matrix).flatten()
        ranked_indices = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])[1:6]

        st.subheader(f"Because you liked **{movie_selected}**, you might enjoy:")
        cols = st.columns(5)
        for idx, (item_idx, score) in enumerate(ranked_indices):
            with cols[idx]:
                st.info(f"**{movie_db.iloc[item_idx]['title']}**")
                st.caption(f"Match Score: {score*100:.1f}%")

# --- PAGE 2: SONG RECOMMENDER ---
else:
    st.title("🎵 Rhythm Recommender")
    st.markdown("Select a track, and our system will run content comparisons across audio characteristics within the same genre profile.")

    song_selected = st.selectbox("Type or select a song:", song_db['title'].values)

    if st.button("Generate Song Recommendations"):
        target_record = song_db[song_db['title'] == song_selected].iloc[0]
        target_genre = target_record['top genre']
        
        genre_scoped_df = song_db[song_db['top genre'] == target_genre].copy()
        metrics_matrix = genre_scoped_df.drop(['title', 'top genre'], axis=1).astype(np.float32)
        target_metrics = target_record.drop(['title', 'top genre']).astype(np.float32)
        
        linear_correlations = metrics_matrix.corrwith(target_metrics, axis=1)
        matched_indices = linear_correlations.sort_values(ascending=False).head(6).index
        
        suggested_tracks = song_db.loc[matched_indices, 'title'].tolist()
        if song_selected in suggested_tracks:
            suggested_tracks.remove(song_selected)

        st.subheader(f"Top matches sharing the **{target_genre}** genre signature:")
        for rank, track in enumerate(suggested_tracks[:5], start=1):
            st.success(f"🎶 **{rank}. {track}**")