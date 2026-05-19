import pickle, numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


#  Progress logger 
# Gensim is quiet by default. 
class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        self.epoch += 1
        loss = model.get_latest_training_loss()
        if self.epoch % 5 == 0:  # print every 5 epochs
            print(f"  Epoch {self.epoch}/50  loss: {loss:.1f}")


def train_action2vec(
    corpus_path = 'data/processed/corpus.pkl',
    model_path  = 'models/action2vec.model',
    vector_size = 128,
    window      = 5,
    min_count   = 1,
    epochs      = 50,
):
    #  1. Load the corpus 
    # corpus is a List[List[str]] ie each inner list is one possession chain
    # e.g. [['PASS_PROG', 'CARRY', 'PASS_PROG', 'DRIBBLE_W', 'SHOT_ON'], ...]
    print("Loading corpus...")
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    # Filter out chains shorter than 2 tokens — they add noise
    corpus = [c for c in corpus if len(c) >= 2]
    print(f"Corpus: {len(corpus):,} chains after filtering")

    #  2. Train ─
    print("Training Action2Vec (this may take a few minutes)...")
    model = Word2Vec(
        sentences   = corpus,
        vector_size = vector_size,  # each token gets a 64-dim vector
        window      = window,        # look 5 actions left and right
        min_count   = min_count,     # ignore tokens seen < min_count times
        workers     = 4,             # parallel CPU threads
        sg          = 1,             # 1 = Skip-gram (better for rare tokens)
        negative    = 10,            # negative sampling to speed up training
        epochs      = epochs,
        seed        = 42,
        compute_loss= True,          # needed for EpochLogger
        callbacks   = [EpochLogger()],
    )

    #  3. Save 
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Vocabulary ({len(model.wv)}): {list(model.wv.key_to_index.keys())}")
    return model



import json


def build_player_embeddings(
    model_path   = 'models/action2vec.model',
    actions_path = 'data/processed/player_actions.pkl',
    min_actions  = 20,   # skip players with fewer than 20 actions
):
    #  1. Load the trained model and player action sequences 
    model = Word2Vec.load(model_path)
    with open(actions_path, 'rb') as f:
        player_actions = pickle.load(f)  # {player_id: ['PASS_PROG', 'CARRY', ...]}

    print(f"Building embeddings for {len(player_actions):,} raw players...")

    #  2. For each player, average their action token vectors 
    # the centroid of their  action vectors in the 64-dim embedding space.
    embeddings = {}
    skipped    = 0

    for player_id, tokens in player_actions.items():
        # Only keep tokens the model knows (all 12 should be present)
        known_vecs = [model.wv[t] for t in tokens if t in model.wv]

        if len(known_vecs) < min_actions:
            skipped += 1
            continue  # too few actions for a reliable embedding

        # np.mean across axis=0 averages the vectors element-wise
        # Result shape: (64,) -> one number per embedding dimension
        embeddings[player_id] = np.mean(known_vecs, axis=0)

    print(f"Built {len(embeddings):,} player embeddings (skipped {skipped} with < {min_actions} actions)")

    #  3. Save as a numpy matrix + a matching list of player IDs 
    # The matrix row i corresponds to player player_ids[i]
    # This pairing is critical to keep them in sync!
    player_ids = list(embeddings.keys())
    matrix     = np.array(list(embeddings.values()))   # shape: (N, 64)

    np.save('models/player_matrix.npy', matrix)
    with open('models/player_ids.json', 'w') as f:
        json.dump(player_ids, f)

    print(f"Saved: player_matrix.npy  shape={matrix.shape}")
    print(f"Saved: player_ids.json   ({len(player_ids)} entries)")
    return player_ids, matrix

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'aggregate':
        build_player_embeddings()
    else:
        train_action2vec()