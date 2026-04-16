# fca — Formal Concept Analysis

This folder contains the statistical analysis pipeline that takes GA results produced by `core/` and extracts structural rules about the 64 SBN architectures using Formal Concept Analysis.

---

## Files

| File | Role |
|------|------|
| `SBN_analysis_v0.ipynb` | Full analysis pipeline: hypercube analysis, epistasis, FCA lattice, regression |

---

## Prerequisites

Run `core/SBN_generator_v0.ipynb` first and make sure the output CSV is present in `data/`.

Install dependencies if needed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Configuration (Shared Setup cell)

Two parameters must be set before running the notebook.

**1. CSV filename** — update `_fname` to match the file produced by the generator:

```python
_fname = 'ga_results_algebraic_pop10_gen90.csv'  # <── change this
```

The notebook looks for the file in `data/` first, then falls back to the current directory.

**2. Discretisation thresholds** — run Step 0 first to inspect the score distribution, then set:

```python
THRESH_TOP   = 13.75  # Fit_top   : score strictly above this value  (~50%, median split)
THRESH_ELITE = 14.5   # Fit_elite : score at or above this value      (~10-15%, top cluster)
```

These thresholds produce two binary target columns used from Step 3 onward:

| Column | Rule | Recommended target |
|--------|------|--------------------|
| `Fit_top` | `Best_Score > THRESH_TOP` | ~50% of architectures (32/64) |
| `Fit_elite` | `Best_Score >= THRESH_ELITE` | 10–15% of architectures (6–10/64) |

Thresholds should be anchored on **natural breaks** in the distribution, not mechanical quantiles. The values above are examples from an `algebraic` fitness run; adjust them for each fitness function.

---

## Notebook walkthrough

| Step | Content |
|------|---------|
| **Shared Setup** | Load CSV, define thresholds, compute `Fit_top` and `Fit_elite` |
| **0** | Score distribution, discretisation check, summary table of all 64 architectures |
| **1** | Hypercube edge analysis — marginal effect ΔCᵢ per constraint across all 32 contexts, stability (Coefficient of Variation), sign heatmap |
| **2** | Epistasis matrix — pairwise interaction ε(Cᵢ, Cⱼ) for all 15 constraint pairs, SNR ranking, distribution of ε across contexts |
| **3** | FCA lattice — Stem Base (Duquenne–Guigues basis) for `Fit_top` and `Fit_elite` contexts, formal context heatmaps, implication rules with support and coverage |
| **4** | Post-lattice validation — rule verification against raw data |
| **5** | Regression model — LASSO feature selection, Walsh–Hadamard spectral decomposition, variance by interaction order, LASSO vs Walsh cross-reference |

---

## Key outputs

**Step 1 — Hypercube edge analysis**

For each constraint Cᵢ, 32 edges of the {0,1}⁶ hypercube are examined. The delta Δ = f(Cᵢ=1) − f(Cᵢ=0) measures the marginal effect of activating a constraint in a given context. The Coefficient of Variation (CV) classifies constraints as stable (CV < 1), moderate (1 ≤ CV < 3), or epistatic (CV ≥ 3).

**Step 2 — Epistasis matrix**

Pairwise epistasis ε(Cᵢ, Cⱼ, ctx) = f(1,1) − f(1,0) − f(0,1) + f(0,0) is computed over 16 contexts per pair. The SNR = |μ(ε)| / σ(ε) ranks the 15 pairs from most to least consistent interaction.

**Step 3 — FCA implication rules**

The Stem Base provides the minimal non-redundant set of implications with confidence = 1.0, of the form:

```
premise  →  conclusion
```

where all architectures satisfying `premise` also satisfy `conclusion` within the `Fit_top` or `Fit_elite` context. Each rule is annotated with its support and coverage.

**Step 5 — Walsh–Hadamard spectrum**

The 63 non-trivial Walsh coefficients f̂(S) decompose the fitness landscape by interaction order (main effects, 2-way, …, 6-way). The epistasis ratio reports the fraction of total variance explained by interactions of order ≥ 2.

---

## Interpreting results

> Results from a single GA run per architecture reflect structural tendencies, not statistical certainties. The GA is stochastic — replicate runs may shift threshold positions and alter which rules appear in the Stem Base.
