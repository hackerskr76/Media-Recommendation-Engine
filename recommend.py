import os
import ast
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Hide background tracking messages
warnings.filterwarnings('ignore')

print("--- Step 1: Loading datasets cleanly ---")

# Verify datasets exist in the current folder
required = ['tmdb_5000_movies.csv', 'tmdb_5000_credits.csv', 'top10s.csv', 'top50contry.csv']
missing = [f for f in required if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Missing required CSV files in this directory: {missing}")

# Processing Movies
df_movies = pd.read_csv('tmdb_5000_movies.csv')
df_credits = pd.read_csv('tmdb_5000_credits.csv')
movies = df_movies.merge(df_credits, on='title')[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

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

def strip_spaces(L): return [i.replace(" ", "") for i in L]
for col in ['genres', 'keywords', 'cast', 'crew']:
    movies[col] = movies[col].apply(strip_spaces)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movie_db = movies[['movie_id', 'title', 'tags']].copy()
movie_db['tags'] = movie_db['tags'].apply(lambda x: " ".join(x).lower())

stemmer = PorterStemmer()
movie_db['tags'] = movie_db['tags'].apply(lambda text: " ".join([stemmer.stem(w) for w in text.split()]))

# Processing Songs
songs_10s = pd.read_csv('top10s.csv', encoding='latin-1')
songs_50s = pd.read_csv('top50contry.csv', encoding='latin-1')
songs_10s = songs_10s.drop(['year', 'artist', 'Unnamed: 0'], axis=1, errors='ignore')
songs_50s = songs_50s.drop(['year', 'artist', 'Unnamed: 0', 'country', 'added'], axis=1, errors='ignore')
song_db = pd.concat([songs_10s, songs_50s], ignore_index=True).drop_duplicates(subset=['title']).reset_index(drop=True)

print("✅ Data cleaning complete!")

# Build text sparse vector space safely
vectorizer = CountVectorizer(max_features=3000, stop_words='english')
movie_sparse_matrix = vectorizer.fit_transform(movie_db['tags'])


# --- CORE ENGINES ---
def recommend_movies(movie_title):
    matching = movie_db[movie_db['title'].str.lower() == movie_title.lower()]
    if matching.empty:
        print(f"❌ Movie '{movie_title}' not found.")
        return
    idx = matching.index[0]
    target_vector = movie_sparse_matrix[idx]
    scores = cosine_similarity(target_vector, movie_sparse_matrix).flatten()
    ranked = sorted(list(enumerate(scores)), reverse=True, key=lambda x: x[1])[1:6]
    
    print(f"\n🎬 Recommendations for '{movie_db.iloc[idx]['title']}':")
    for r, (item_idx, score) in enumerate(ranked, start=1):
        print(f" {r}. {movie_db.iloc[item_idx]['title']} ({score*100:.1f}% match)")

def recommend_songs(song_title):
    matching = song_db[song_db['title'].str.lower() == song_title.lower()]
    if matching.empty:
        print(f"❌ Song '{song_title}' not found.")
        return
    record = matching.iloc[0]
    genre = record['top genre']
    
    genre_df = song_db[song_db['top genre'] == genre].copy()
    matrix = genre_df.drop(['title', 'top genre'], axis=1).astype(np.float32)
    target = record.drop(['title', 'top genre']).astype(np.float32)
    
    correlations = matrix.corrwith(target, axis=1)
    indices = correlations.sort_values(ascending=False).head(6).index
    tracks = song_db.loc[indices, 'title'].tolist()
    if record['title'] in tracks:
        tracks.remove(record['title'])
        
    print(f"\n🎵 Tracks matching the [{genre}] profile of '{record['title']}':")
    for r, track in enumerate(tracks[:5], start=1):
        print(f" {r}. {track}")


# --- RUN TEST DEMOS ---
print("\n--- Step 2: Testing Recommendations ---")
recommend_movies("Avatar")
recommend_movies("The Dark Knight Rises")

recommend_songs("Bad Romance")