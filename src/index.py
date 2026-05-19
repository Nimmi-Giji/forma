from pathlib import Path
import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MATRIX_PATH = MODELS_DIR / "player_matrix.npy"
INDEX_PATH = MODELS_DIR / "faiss_index.bin"


def build_index():
    matrix = np.load(MATRIX_PATH).astype("float32")

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_PATH))

    print(f"Index built: {index.ntotal} players, dim={index.d}")


def find_similar(player_idx, top_k=10):
    index = faiss.read_index(str(INDEX_PATH))
    matrix = np.load(MATRIX_PATH).astype("float32")

    query = matrix[player_idx].reshape(1, -1)

    distances, indices = index.search(query, top_k + 1)

    # FAISS already returns squared distances
    return list(zip(indices[0][1:], distances[0][1:]))


if __name__ == "__main__":
    build_index()