# 🎬 Dynamic Media Recommendation Engine 🎵

A production-ready, memory-optimized Python application that delivers high-accuracy content recommendations for both movies and music. This project combines **Natural Language Processing (NLP)** for text-based movie similarity and **Statistical Feature Correlation** for audio-metric song matching into a unified, high-performance architecture.

---

## 🚀 Key Upgrades & Performance Optimizations
The original development notebooks suffered from performance bottlenecks and memory leaks (causing local kernel crashes). This upgraded version introduces critical engineering fixes:
* **Memory Optimization (CSR Matrices):** Instead of storing a massive $N \times N$ dense matrix in RAM which triggers hardware timeouts, text vector spaces are maintained as highly compressed sparse row matrices. Similarity is evaluated *dynamically on the fly* for a single target call stack.
* **Algorithmic Fixes:** Corrected tokenized stemming bugs that previously merged separate descriptors into unreadable strings.
* **Explicit Downcasting:** Enforced numerical metric arrays to process explicitly as `float32` arrays, bypassing thread-level prediction halts during horizontal correlation loops.

---

## 🧠 System Architectures

### 1. Movie Matcher Engine (Content-Based NLP)
* **Dataset:** TMDB 5000 Movies & Credits
* **The Logic:** Evaluates metadata footprints including plots (`overviews`), `genres`, keyword associations, `cast` members, and `directors`.
* **The Math:** Words are tokenized and processed via a Porter Stemmer to reduce vocabulary inflection. Text matrices are vector-mapped using a 3,000-feature `CountVectorizer` (with English stop-word exclusions). Distances are measured using **Cosine Similarity**:

$$\text{Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

### 2. Rhythm Recommender Engine (Feature Correlation Matching)
* **Dataset:** Spotify Top 10s & Country Hits
* **The Logic:** Matches tracks according to acoustic signatures rather than text tags. It slices the matrix down to the exact match space of the target track's `top genre` to maintain generic consistency.
* **The Math:** Computes a **Pearson Correlation Matrix** across key structural numeric features: *BPM (Tempo), Energy, Danceability, Valence (Positivity), and Acousticness*. It isolates rows showing the highest proportional alignment to the input track's coordinate footprint.

---

## 📦 Project Directory Structure
```text
📦 Media-Recommendation-Engine
 ┣ 📜 app.py               # Interactive Streamlit Web UI Dashboard
 ┣ 📜 recommend.py         # Memory-safe, high-speed terminal execution core
 ┣ 📜 requirements.txt     # Complete operational environment dependencies
 ┣ 📜 .gitignore           # Keeps data sets and caches out of your repository
 ┗ 📜 README.md            # Project technical presentation profile
🛠️ Installation & Local Setup
1. Clone the Workspace
Bash
git clone [https://github.com/YOUR_USERNAME/Media-Recommendation-Engine.git](https://github.com/YOUR_USERNAME/Media-Recommendation-Engine.git)
cd Media-Recommendation-Engine
2. Configure Environment & Dependencies
Ensure your dataset files (tmdb_5000_movies.csv, tmdb_5000_credits.csv, top10s.csv, top50contry.csv) are placed directly in the root folder.

Bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\Activate.ps1

# Install required numerical, NLP, and UI libraries
pip install -r requirements.txt
💻 Operational Usages
Terminal Testing Check
To quickly test accuracy and verification scores inside your command window, execute the main python script:

Bash
python recommend.py
Launching the Graphical Web Dashboard
To transition your experience into a full-scale interactive web dashboard, run the Streamlit front-end module:

Bash
streamlit run app.py
🎯 Verification Previews
Plaintext
🎬 Recommendations for 'The Dark Knight Rises':
 1. The Dark Knight (41.9% match)
 2. Batman Returns (33.3% match)
 3. Batman (32.3% match)

🎵 Tracks matching the [dance pop] profile of 'Bad Romance':
 1. Good Life
 2. Come & Get It
 3. No Vacancy