import pickle

with open('data/processed/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

print(f"Number of chains: {len(corpus)}")          # expect 50,000+
print(f"Example chain: {corpus[0]}")               # e.g. ['PASS_PROG', 'CARRY', 'SHOT_ON']
print(f"Average chain length: {sum(len(c) for c in corpus)/len(corpus):.1f}")