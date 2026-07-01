# Head Atlas: Developmental Sequence of Attention Head Specialization

**A comprehensive map of when and how attention heads specialize during transformer training.**

## The Gap

Every existing study of attention head specialization probes ONE behavior type in isolation:

| Paper | What they tracked | What they missed |
|-------|------------------|-----------------|
| Olsson et al. 2022 | Induction heads emerge at step X | Everything else |
| Gu et al. 2025 | Dormancy emerges at ~55B tokens | What the non-dormant heads are doing |
| Wang et al. 2025 | Head differentiation via learning coefficient | Only a 2-layer toy model |
| Blackwell 2026 (ours) | Delimiter heads by step ~1000 | Other specialization types |

Nobody has produced a **full developmental atlas**: tracking ALL head behaviors simultaneously from step 0 to convergence in a realistic model. We don't know:

- What develops first? Positional heads before induction? Delimiter before content?
- Is there a fixed developmental order or does it depend on data/tokenizer?
- Do heads transition between types during training?
- When do heads "commit" to a specialization (irreversibly)?
- Does the tokenizer change the developmental sequence (not just the outcome)?

## The Project

Train a single 410M transformer, checkpoint frequently, and at each checkpoint classify every head across multiple behavior types simultaneously.

### Behaviors to probe at each checkpoint

| Behavior | How to measure | Expected emergence |
|----------|---------------|-------------------|
| **Positional (previous token)** | Attention mass on position n-1 | Very early (step 0-100) |
| **Positional (first token / P0 sink)** | Attention mass on position 0 | Early (Gu: by 55B tokens) |
| **Induction** | Copy score: does head attend to token after previous occurrence? | Early-mid (Olsson: phase change) |
| **Delimiter/structural** | Attention mass on delimiter token positions | Early (our data: step ~1000) |
| **Syntactic (bracket matching)** | Attention from close-bracket to open-bracket | Mid? |
| **Content (semantic similarity)** | Correlation between attention and embedding similarity | Mid-late? |
| **Duplicate token** | Attention to previous occurrences of same token | Unknown |
| **Dormant (attention sink)** | HONOR metric < threshold (Sandoval-Segura 2025) | Early, then stabilizes |

### Training setup

- Architecture: GPT-NeoX 410M (same scale as merge-barriers run-002)
- Corpus: **SlimPajama** or **RedPajama-v2** (standard pretraining mix approximating production models: ~70% web, ~10% code, ~5% academic, ~5% books, ~5% Wikipedia, ~5% misc). NOT the structured-data-heavy corpus from merge-barriers experiments. Ensures findings generalize to how production models develop.
- Tokenizer: standard BPE (GPT-NeoX default or similar)
- Checkpoints: every 50 steps for first 2000, every 200 steps to 20000
- Probe data: fixed set of texts covering all behavior types (structured data, code, prose, brackets, repeated tokens)
- **Baseline** (primary): standard tokenizer, standard corpus. The "normal" developmental atlas.
- **Comparison** (second run): merge-barrier tokenizer, same corpus. Tests whether the tokenizer changes the developmental SEQUENCE, not just the outcome.

### Output

For each checkpoint, for each head (24 layers x 16 heads = 384):
- Score on each of 8 behavior types
- Classification into dominant type
- Confidence (how specialized vs generalist)

### Visualizations

- **Developmental timeline**: x=training step, y=head index, color=dominant behavior type. Shows when each head commits.
- **Phase transitions**: when do groups of heads shift behavior simultaneously?
- **Tokenizer comparison**: same timeline for Model A vs Model B. Do heads specialize in the same order?
- **Layer depth analysis**: do early layers specialize first, or is it distributed?

## Why This Matters

1. **Developmental interpretability** is a young subfield. This would be the first comprehensive atlas at realistic scale.
2. **Training efficiency**: if we know WHEN heads commit, we could intervene (change learning rate, inject data) at critical periods.
3. **Extends our merge-barriers work**: the tokenizer changes the outcome (which heads develop). Does it also change the SEQUENCE?
4. **Practical for model providers**: understanding head development could inform curriculum learning, data mixing schedules, and early stopping decisions.

## Relationship to Other Work

This project emerges from the merge-barriers research (github.com/blackwell-systems/merge-barriers). During that work, we measured delimiter head emergence at step ~1000 and gradient-attention coupling over 20K steps. The natural next question: what ELSE is developing simultaneously?

## Status

Planning. Requires:
- [ ] Finalize probe set (texts that test each behavior type)
- [ ] Write multi-probe evaluation script
- [ ] Train with frequent checkpointing
- [ ] Analysis and visualization

Estimated cost: ~$5-10 (one 410M training run on a 4090, ~2 hours, plus probe evaluation at each checkpoint).

## Infrastructure

- Training: adapted `train_model.py` from structok repo (frequent checkpointing mode)
- Hardware: RTX 4090 or A100 on Vast.ai
- Data: SlimPajama or RedPajama-v2 (~5GB sample, standard pretraining distribution)
- Tokenizer (baseline): standard BPE 64K vocab (no barriers)
- Tokenizer (comparison): structok-64k.json (merge barriers)
- Probes: custom multi-behavior evaluation script (new, to be written)
