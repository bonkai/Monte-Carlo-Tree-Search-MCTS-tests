# MCTS for LLM Response Generation

An experiment that applies **Monte Carlo Tree Search** to language-model decoding: instead
of taking the first answer a model gives, it treats response generation as a search tree,
explores multiple candidate branches (varying temperature and prompt additions), scores
them, and converges on the strongest answer.

## How it works

- **Search tree** — each node is a candidate response with its own temperature and prompt
  variation (`mcts.py`, `MCTSNode`).
- **UCB selection** — nodes are chosen by an Upper Confidence Bound score, balancing
  exploitation of promising branches against exploration of new ones.
- **Memory** — a context manager + vector store (`memory_manager.py`) retain relevant
  state across the search.
- **Persistence** — runs are logged and stored via `database_manager.py`.
- Includes a baseline `nomcts.py` for comparison and a `chatbot_ui.py` front end.

## Stack

- Python, NumPy, a local LLM endpoint (Ollama-compatible), SQLite-backed storage

## Run

```bash
pip install numpy requests
python main.py
```

> Research/experimental code — a testbed for search-guided decoding, not a production library.
