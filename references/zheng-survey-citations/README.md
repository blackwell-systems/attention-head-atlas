# Zheng et al. Survey Citations

Papers cited by Zheng et al. (2024) "Attention Heads of Large Language Models: A Survey" (Patterns, Feb 2025). Organized by the survey's four-stage taxonomy. 48 papers downloaded from arXiv.

## Knowledge Recalling (KR)

Memory heads, Constant heads, Single Letter heads, Negative heads.

| File | Paper | Year | Head types |
|------|-------|------|------------|
| dana-2024-memorization-attention-only.pdf | Memorization in Attention-Only Transformers | 2024 | Memory |
| yao-2024-knowledge-circuits.pdf | Knowledge Circuits in Pretrained Transformers | 2024 | Knowledge/Memory |
| lv-2024-factual-recall-mechanisms.pdf | Interpreting Key Mechanisms of Factual Recall | 2024 | Memory/Factual |
| yu-2024-correcting-negative-bias.pdf | Correcting Negative Bias in LLMs | 2024 | Negative |

## In-Context Identification (ICI)

Previous/Positional heads, Rare Words, Duplicate, Subword Merge, Syntactic, Name Mover, Letter Mover, Context, Content Gatherer, Semantic Induction, Subject, Relation.

| File | Paper | Year | Head types |
|------|-------|------|------------|
| wu-2024-retrieval-head.pdf | Retrieval Head Mechanistically Explains Long-Context Factuality | 2024 | Retrieval/Positional |
| tang-2024-razorattention.pdf | RazorAttention: Efficient KV Cache Compression | 2024 | Positional/Retrieval |
| mcdougall-2023-copy-suppression.pdf | Copy Suppression: Comprehensively Understanding an Attention Head | 2023 | Duplicate/Copy |
| tigges-2023-sentiment-representations.pdf | Linear Representations of Sentiment in LLMs | 2023 | Semantic/Sentiment |
| chughtai-2024-summing-up-facts.pdf | Summing Up the Facts: Additive Mechanisms Behind Factual Recall | 2024 | Subject/Relation |
| ferrando-2024-information-flow-routes.pdf | Information Flow Routes: Automatically Interpreting LMs | 2024 | Context/Information flow |
| jin-2024-concept-depth.pdf | Concept Depth: How LLMs Acquire New Concepts | 2024 | Content/Semantic |
| li-2024-geometry-of-concepts.pdf | The Geometry of Concepts: Sparse Autoencoder Feature Structure | 2024 | Semantic/Content |

## Latent Reasoning (LR)

Induction heads, Function Vector, Truthfulness, Accuracy, Consistency, Inhibition.

| File | Paper | Year | Head types |
|------|-------|------|------------|
| olsson-2022-induction-heads.pdf | In-Context Learning and Induction Heads | 2022 | Induction |
| edelman-2024-statistical-induction-heads.pdf | The Evolution of Statistical Induction Heads | 2024 | Induction |
| crosbie-2024-induction-heads-essential.pdf | Are Induction Heads Essential for In-Context Learning? | 2024 | Induction |
| ji-an-2024-icl-episodic-memory.pdf | ICL as Episodic Memory Retrieval | 2024 | Induction/ICL |
| liang-2024-internal-consistency.pdf | Internal Consistency and Self-Feedback in LLMs | 2024 | Consistency/Truthfulness |
| li-2024-look-within-hallucinate.pdf | Look Within, Why LLMs Hallucinate | 2024 | Truthfulness |
| chuang-2024-lookback-lens.pdf | Lookback Lens for Detecting and Mitigating Contextual Hallucinations | 2024 | Truthfulness |
| ji-2024-internal-states-hallucination.pdf | LLM Internal States Reveal Hallucination Risk | 2024 | Truthfulness/Accuracy |
| kim-2024-syllogistic-reasoning.pdf | Syllogistic Reasoning in LLMs | 2024 | Reasoning |
| wiegreffe-2024-answer-assemble-ace.pdf | Answer, Assemble, ACE: Understanding How Transformers Answer Multiple Choice | 2024 | Accuracy/Correct Letter |
| lieberum-2023-circuit-analysis-scale.pdf | Does Circuit Analysis Interpretability Scale? | 2023 | Circuits/General LR |

## Expression Preparation (EP)

Mixed, Amplification, Correct, Coherence, Faithfulness.

| File | Paper | Year | Head types |
|------|-------|------|------------|
| liang-2024-controllable-text-generation.pdf | Controllable Text Generation via Activation Steering | 2024 | Coherence/Steering |
| turner-2023-activation-addition.pdf | Activation Addition: Steering LLMs Without Optimization | 2023 | Amplification/Steering |
| fu-2024-moa-sparse-attention.pdf | Mixture of Attention: Sparse Attention Patterns | 2024 | Mixed/Sparse |

## Methodology

Papers about experimental methods: activation patching, probing, ablation, logit lens.

| File | Paper | Year | Method |
|------|-------|------|--------|
| heimersheim-2024-activation-patching.pdf | How to Use and Interpret Activation Patching | 2024 | Activation patching |
| hoscilowicz-2024-ni-iii-probing.pdf | Non-Invasive Probing Methods | 2024 | Probing |
| yin-2024-loft-fine-tuning.pdf | LoFT: Local Fine-Tuning | 2024 | Fine-tuning methodology |
| yu-2024-xfinder.pdf | xFinder: Robust and Pinpointed Answer Extraction | 2024 | Evaluation |
| yu-2024-two-towers-metric-learning.pdf | Two Towers Metric Learning for Attention | 2024 | Metric learning |

## General / Foundational

Transformer architecture, scaling, surveys, cognitive science connections.

| File | Paper | Year | Topic |
|------|-------|------|-------|
| kaplan-2020-scaling-laws.pdf | Scaling Laws for Neural Language Models | 2020 | Scaling |
| bommasani-2021-foundation-models.pdf | On the Opportunities and Risks of Foundation Models | 2021 | Survey |
| devlin-2018-bert.pdf | BERT: Pre-training of Deep Bidirectional Transformers | 2018 | Architecture |
| shazeer-2020-glu-variants.pdf | GLU Variants Improve Transformer | 2020 | Architecture |
| santana-2021-neural-attention-survey.pdf | Neural Attention: A Survey | 2021 | Attention survey |
| luo-2024-understanding-to-utilization.pdf | From Understanding to Utilization: A Survey on Explainability | 2024 | Interpretability survey |
| dasgupta-2022-human-like-content-effects.pdf | Language Models Show Human-Like Content Effects | 2022 | Cognitive science |
| janik-2023-human-memory-llms.pdf | Aspects of Human Memory and LLMs | 2023 | Cognitive science |
| mischler-2024-contextual-feature-extraction.pdf | Contextual Feature Extraction Hierarchies | 2024 | Cognitive science |
| millidge-2021-predictive-coding.pdf | Predictive Coding: A Theoretical and Experimental Review | 2021 | Cognitive science |
| whitehill-2013-understanding-act-r.pdf | Understanding ACT-R | 2013 | Cognitive science |
| laird-2022-act-r-soar.pdf | ACT-R and Soar | 2022 | Cognitive science |
| hagendorff-2023-machine-psychology.pdf | Machine Psychology | 2023 | Cognitive science |
| johansson-2024-functional-equivalence.pdf | Functional Equivalence of LLMs and Cognitive Models | 2024 | Cognitive science |
| hendrycks-2021-math-dataset.pdf | Measuring Mathematical Problem Solving (MATH) | 2021 | Benchmark |
| cobbe-2021-training-verifiers.pdf | Training Verifiers to Solve Math Word Problems (GSM8K) | 2021 | Benchmark |
| narayan-2018-extreme-summarization.pdf | Don't Give Me the Details, Just the Summary! (XSum) | 2018 | Benchmark |

## Notable gap

The survey's taxonomy of ~30 head types does not include **spacing heads** (whitespace boundary recovery), which our research shows is the single largest category: 40-48% of all heads across two architectures. This category was invisible until spacing was added to the probe taxonomy (Blackwell, 2026c).

## Note

48 papers downloaded from arXiv. ~40 additional references from the survey were conference proceedings, books, or journal articles without arXiv IDs and are not included here. PDFs are gitignored.
