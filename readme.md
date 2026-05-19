# Forma

Player style embeddings from football event sequences.

Forma treats a player's career as a corpus of possession chains -- the same way
Word2Vec treats a document as a corpus of sentences. Each action (progressive pass,
carry, dribble, shot, press) is a token. Word2Vec learns which tokens appear in
similar contexts, producing a 128-dimensional action embedding. Players are then
represented as the weighted mean of their action vectors, clustered with HDBSCAN,
and made searchable via a FAISS similarity index.


## What it does

- Maps every player in La Liga (2012-2021) to a point in style space
- Discovers player archetypes automatically (no predefined categories)
- Finds the N most stylistically similar players to any query player
- Serves an interactive dashboard: style map, similarity search, archetype explorer

Notable finding: the model discovers that team system dominates individual style
signals. Barcelona and Real Madrid players cluster together regardless of position
because their high-possession systems produce near-identical action token
distributions across all roles.


## Stack

| Layer | Tool |
|---|---|
| Data | StatsBombPy (free open data) |
| Embeddings | Gensim Word2Vec (skip-gram, 128D) |
| Dimensionality reduction | UMAP |
| Clustering | HDBSCAN |
| Similarity index | FAISS (IndexFlatL2) |
| Dashboard | Streamlit + Plotly |


## Project structure

    forma/
    |-- app/
    |   `-- dashboard.py       # Streamlit app
    |-- data/
    |   |-- raw/               # StatsBomb downloads (gitignored)
    |   `-- processed/
    |       `-- player_meta_aligned.csv
    |-- models/                # Trained artifacts (gitignored, rebuild with pipeline)
    |-- notebooks/
    |   |-- 01_eda.ipynb
    |   |-- 02_tokenisation.ipynb
    |   |-- 03_embeddings.ipynb
    |   |-- 04_clustering.ipynb
    |   `-- 05_similarity.ipynb
    |-- src/
    |   |-- tokenise.py        # 17-token action vocabulary
    |   |-- chains.py          # Possession chain segmentation
    |   |-- embed.py           # Action2Vec training + player embedding
    |   |-- reduce.py          # UMAP + HDBSCAN
    |   `-- index.py           # FAISS index build + query
    |-- tests/
    |   `-- test_tokenise.py
    |-- requirements.txt
    `-- README.md


## Quick start

    git clone https://github.com/Nimmi-Giji/forma
    cd forma
    python -m venv venv

    # Mac/Linux
    source venv/bin/activate
    # Windows
    venv\Scripts\activate

    pip install -r requirements.txt

Run the full pipeline (takes 20-40 minutes depending on machine):

    python -m src.embed          # download data, tokenise, train Word2Vec, build player embeddings
    python -m src.reduce         # UMAP + HDBSCAN clustering
    python -m src.index          # build FAISS index

Launch the dashboard:

    streamlit run app/dashboard.py

Open http://localhost:8501 in your browser.


## Rebuild from scratch

If you want to regenerate everything from the StatsBomb raw data:

1. Run notebooks 01 through 05 in order
2. Or run the three pipeline commands above
3. player_meta_aligned.csv is the only processed file committed to the repo along with lookup files


## Limitations

The 17-token vocabulary captures carry subtypes (CARRY, CARRY_PROG, CARRY_WIDE,
CARRY_WIDE_PROG), pass types, dribble outcomes, shot location (in/out of box),
press, interception, clearance, and foul won. Token frequencies vary significantly: CARRY leads at 191,000 occurrences while SHOT_ON appears only 1,200 times,
meaning shot-related embeddings are noisier than carry or pass embeddings. Players
with low shot volume (defenders, holding midfielders) are therefore better
represented than attackers whose style is defined by finishing.

The model also cannot separate players who perform the same actions in different
contexts. A deep-lying midfielder who passes progressively looks identical to an
attacking midfielder who does the same. Possible improvements:

- Frequency-weight tokens by rarity to reduce CARRY dominance
- Zone-aware pass tokens (PASS_PROG_OWN_HALF vs PASS_PROG_OPP_HALF)
- Richer source data from StatsBomb 360 freeze frames

## Data

StatsBomb open data is available at https://github.com/statsbomb/open-data under
the StatsBomb Open Data Licence. No account or API key is required.