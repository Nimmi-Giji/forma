import pandas as pd
from src.tokenise import tokenise_event

def build_corpus(events_df):
    """
    Converts a match's events DataFrame into:
      1. corpus         - List[List[str]]: possession chains (our 'sentences')
      2. player_actions - dict: { player_id: [token, token, ...] }

    How possession chains work:
      - A new chain starts whenever the ball changes team
      - Events within a chain are in time order (StatsBomb preserves this)
      - We only include events that return a token (not None)
    """
    corpus         = []
    player_actions = {}
    current_chain  = []
    current_team   = None

    for _, row in events_df.iterrows():
        row_dict  = row.to_dict()
        team      = row_dict.get('team')
        player_id = row_dict.get('player_id')

        # StatsBomb team can be a dict {'id':..., 'name':...} or just a string
        # Normalise it to a consistent value for comparison
        if isinstance(team, dict):
            team = team.get('id')

        # Detect possession change 
        if team != current_team and team is not None:
            # Save the chain we just finished (if it has 2+ tokens)
            if len(current_chain) >= 2:
                corpus.append(current_chain)
            current_chain = []
            current_team  = team

        # Tokenise this event 
        token = tokenise_event(row_dict)

        if token:
            current_chain.append(token)

            # Track per-player actions for building player embeddings later
            if player_id:
                if isinstance(player_id, dict):
                    player_id = player_id.get('id')
                if player_id not in player_actions:
                    player_actions[player_id] = []
                player_actions[player_id].append(token)

    # the last chain
    if len(current_chain) >= 2:
        corpus.append(current_chain)

    return corpus, player_actions


def corpus_stats(corpus):
    """Print some useful stats about your corpus."""
    lengths = [len(c) for c in corpus]
    print(f"Total chains:       {len(corpus):,}")
    print(f"Average chain len:  {sum(lengths)/len(lengths):.1f} tokens")
    print(f"Longest chain:      {max(lengths)} tokens")
    print(f"Shortest chain:     {min(lengths)} tokens")

    # Token frequency across all chains
    from collections import Counter
    all_tokens = [t for chain in corpus for t in chain]
    freq = Counter(all_tokens)
    print(f"\nToken frequencies:")
    for token, count in freq.most_common():
        print(f"  {token:<15} {count:>8,}")