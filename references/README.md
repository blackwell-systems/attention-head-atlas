# Reference Library

Papers relevant to the developmental atlas research program. Organized by topic.

## Foundations

| File | Paper | Year |
|------|-------|------|
| vaswani-2017-attention-is-all-you-need.pdf | Attention Is All You Need | 2017 |
| sennrich-2016-bpe-subword.pdf | Neural Machine Translation of Rare Words with Subword Units (BPE) | 2016 |
| kudo-2018-sentencepiece.pdf | SentencePiece: A Simple and Language Independent Subword Tokenizer | 2018 |
| kaplan-2020-scaling-laws.pdf | Scaling Laws for Neural Language Models | 2020 |
| hoffmann-2022-chinchilla.pdf | Training Compute-Optimal Large Language Models (Chinchilla) | 2022 |

## Mechanistic Interpretability

| File | Paper | Year |
|------|-------|------|
| olah-2020-zoom-in.pdf | Zoom In: An Introduction to Circuits | 2020 |
| elhage-2022-mathematical-framework-circuits.pdf | A Mathematical Framework for Transformer Circuits | 2022 |
| elhage-2022-toy-models-superposition.pdf | Toy Models of Superposition | 2022 |
| olsson-2022-induction-heads.pdf | In-Context Learning and Induction Heads | 2022 |
| conmy-2023-automated-circuit-discovery.pdf | Towards Automated Circuit Discovery for Mechanistic Interpretability | 2023 |
| wang-2023-interpretability-wild-ioi.pdf | Interpretability in the Wild: IOI Circuit | 2023 |
| bricken-2023-towards-monosemanticity.pdf | Towards Monosemanticity (Sparse Autoencoders) | 2023 |
| cunningham-2024-sparse-autoencoders.pdf | Sparse Autoencoders Find Highly Interpretable Features | 2024 |
| syed-2024-attribution-patching.pdf | Attribution Patching | 2024 |

## Attention Head Specialization

| File | Paper | Year |
|------|-------|------|
| clark-2019-what-does-bert-look-at.pdf | What Does BERT Look At? | 2019 |
| voita-2019-specialized-heads.pdf | Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting | 2019 |
| michel-2019-sixteen-heads.pdf | Are Sixteen Heads Really Better Than One? | 2019 |
| xiao-2024-attention-sinks-streaming.pdf | Efficient Streaming Language Models with Attention Sinks | 2024 |
| gu-2025-when-attention-sink-emerges.pdf | When Attention Sink Emerges in Language Models (ICLR 2025) | 2025 |
| sandoval-segura-2025-dormant-heads.pdf | Active-Dormant Attention Heads (arXiv:2410.13835) | 2025 |

## Developmental Interpretability

| File | Paper | Year |
|------|-------|------|
| wang-2025-head-differentiation.pdf | Differentiation and Specialization of Attention Heads via rLLC (ICLR 2025 Spotlight) | 2025 |
| wang-2025-embryology-lm.pdf | Embryology of a Language Model (arXiv:2508.00331) | 2025 |
| wang-2025-patterning.pdf | Patterning: The Dual of Interpretability (arXiv:2601.13548) | 2026 |
| start-making-senses.pdf | Start Making Sense(s): Tracking When Attention Heads Specialize on Word Senses | 2025 |
| predicting-induction-heads.pdf | Predicting the Emergence of Induction Heads (arXiv:2511.16893) | 2026 |
| baherwani-2026-emergent-stochastic.pdf | Emergent Capabilities Arise Randomly (arXiv:2606.25010) | 2026 |
| xu-2026-attention-circuits-form.pdf | When Do Attention Circuits Form? (arXiv:2606.02378) | 2026 |
| nanda-2023-progress-measures-grokking.pdf | Progress Measures for Grokking via Mechanistic Interpretability | 2023 |
| biderman-2023-pythia.pdf | Pythia: A Suite for Analyzing Large Language Models Across Training | 2023 |

## Singular Learning Theory

| File | Paper | Year |
|------|-------|------|
| wei-2022-deep-learning-singular.pdf | Deep Learning is Singular, and That's Good | 2022 |
| baker-2025-structural-inference.pdf | Structural Inference: Interpreting Small Language Models with Susceptibilities | 2025 |
| lau-2025-llc-rllc.pdf | The Local Learning Coefficient (practical SLT) | 2025 |

## Surveys and Infrastructure

| File | Paper | Year |
|------|-------|------|
| rogers-2021-bertology.pdf | A Primer in BERTology | 2021 |
| schmidt-2024-tokenization-survey.pdf | Tokenization Survey | 2024 |
| men-2024-shortgpt-layer-pruning.pdf | ShortGPT: Layers in Large Language Models Are More Redundant Than You Expect | 2024 |
| musat-2026-phase-transitions-copy.pdf | Phase Transitions in Copy Tasks | 2026 |

## Reading Order

For someone entering this field from the experimental side:

1. Vaswani (2017), Sennrich (2016): understand the architecture and tokenizer you're studying
2. Olah (2020), Elhage (2022) framework: visual and mathematical foundations for circuits
3. Olsson (2022), Voita (2019), Michel (2019): the attention head specialization literature
4. Gu (2025), Sandoval-Segura (2025), Xiao (2024): attention sinks and dormancy
5. Wang (2025a), Wang (2025b): developmental interpretability and the spacing fin
6. Wang & Murfet (2026), Baker (2025), Wei (2022): singular learning theory and patterning
7. Everything else as needed

## Tokenization Effects

| File | Paper | Year |
|------|-------|------|
| alqahtani-2025-stop-taking-tokenizers.pdf | Stop Taking Tokenizers for Granted (EACL 2026) | 2025 |
| litetoken-2026-merge-residues.pdf | LiteToken: Removing Intermediate Merge Residues from BPE | 2026 |
| lian-2024-scaffold-bpe.pdf | Scaffold-BPE: Enhancing BPE with Scaffold Token Removal (AAAI 2025) | 2024 |
| meta-2024-byte-latent-transformer.pdf | Byte Latent Transformer: Patches Scale Better Than Tokens | 2024 |
| liu-2025-superbpe.pdf | SuperBPE: Space Travel for Language Models (COLM 2025) | 2025 |
| land-2025-bpe-script.pdf | BPE Stays on SCRIPT: Structured Encoding for Multilingual Pretokenization (ICML 2025 Best Paper) | 2025 |
| chizhov-2026-source-attributed-bpe.pdf | From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution | 2026 |
| karim-2025-structured-data-tokenisation.pdf | Innovative Tokenisation of Structured Data for LLM Training | 2025 |
| kutschka-2026-notation-matters.pdf | Notation Matters: A Benchmark Study of Token-Optimized Formats in Agentic AI Systems | 2026 |
| ayoobi-2026-tokenizer-betrays-reasoning.pdf | Say Anything but This: When Tokenizer Betrays Reasoning in LLMs | 2026 |

## Head Specialization at Scale

| File | Paper | Year |
|------|-------|------|
| zheng-2024-attention-heads-survey.pdf | Attention Heads of Large Language Models: A Survey (Patterns 2025) | 2024 |
| schallon-2026-collapsed-heads-alibi.pdf | Surgical Repair of Collapsed Attention Heads in ALiBi Transformers | 2026 |
| kaplan-2025-from-tokens-to-words.pdf | From Tokens to Words: On the Inner Lexicon of LLMs (ICLR 2025) | 2025 |

## Training Dynamics and Circuit Formation

| File | Paper | Year |
|------|-------|------|
| hoogland-2024-developmental-landscape-icl.pdf | The Developmental Landscape of In-Context Learning (Timaeus) | 2024 |
| chen-2024-provable-induction-heads.pdf | Unveiling Induction Heads: Provable Training Dynamics (NeurIPS 2024) | 2024 |
| incremental-2025-sparse-attention.pdf | Incremental Learning of Sparse Attention Patterns in Transformers | 2025 |
| odonnat-2024-mechanistic-training-dynamics.pdf | A Mechanistic Study of Transformers Training Dynamics | 2024 |
| bayazit-2025-crosscoding-through-time.pdf | Crosscoding Through Time: Tracking Feature Emergence | 2025 |
| ge-2025-concept-evolution-pretraining.pdf | Evolution of Concepts in Language Model Pre-Training (ICLR 2026) | 2025 |
| jain-2026-retokenization-symmetry.pdf | Emergent Retokenization Symmetry in Large Language Models | 2026 |

## Note

57 papers total. PDFs are gitignored. This README is committed; the papers are local only.
