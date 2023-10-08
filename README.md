This repo contains experimentation geared towards reducing the memory footprint of transformer decoder KV cache through pruning/consolidating its KV vectors across attention heads. Experimentation is currently limited to GPT-2 on WikiText.

Experiments:
- [K-Sim Analysis.ipynb](<K-Sim Analysis.ipynb>) - Evaluate consolidation of attention KVs for which key vectors have high cosine-similarity
- [A-Sim Analysis.ipynb](<A-Sim Analysis.ipynb>) - Evaluate consolidation of attention KVs for which attention patterns over initial query set show high similarity
- [Low-Rank Analysis.ipynb](<Low-Rank Analysis.ipynb>) - Analyze rank and singular value density of attention matrices. Examine perplexity under low-rank approximations of the attention matrix to motivate future work.