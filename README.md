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

