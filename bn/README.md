# bn_analysis — Bayesian Network Analysis

This folder contains a two-notebook pipeline that tests Shannon's causal hypothesis
on SBN circuits: do confusion and diffusion fully mediate the relationship between
architectural design decisions and cryptographic performance?

---

## Files

| File | Role |
|------|------|
| `SBN_BN_sampler_v0.ipynb` | Generates multi-run replication data (30 GA runs × 32 architectures) |
| `SBN_BN_analysis_v0.ipynb` | 4-layer Bayesian Network: structure learning, CPT estimation, causal inference |

Run them **in this order**. The analysis notebook reads the CSV files produced by the sampler.

---

## Prerequisites

Run `core/SBN_generator_v0.ipynb` first to produce at least one GA result CSV in `data/`.
The sampler uses it as a convergence reference (Step 5).

Install dependencies:

```bash
pip install numpy pandas matplotlib pgmpy          # sampler
pip install numpy pandas matplotlib networkx scipy bottleneck  # analysis
```

---

## Why a separate sampler?

The standard GA CSV (one best score per architecture) cannot be used directly for
Bayesian Network learning. Two fundamental reasons:

1. **No statistical testability.** Conditional independence tests require a
   *distribution* of scores per architecture, not a single value. One observation
   per cell collapses every CPT entry to 0 or 1 — equivalent to a truth table,
   which FCA already covers more rigorously.

2. **Orthogonal design precludes inter-constraint inference.** The full factorial
   2⁵ design makes A…L perfectly uncorrelated by construction. Any apparent
   association between constraints in the single-point CSV is an artefact of the
   experimental layout.

The sampler solves both problems by running the GA **30 independent times** per
architecture, producing a genuine score distribution per cell.

---

## Why S is fixed to 0

Constraint S (alternating nonlinear/linear layers) directly encodes Shannon's
mechanisms: when S=1, `nl_fraction` is mechanically fixed at 0.600 with zero
variance. Including S in the analysis makes `nl_fraction` non-informative as a
mediator and collapses the causal chain into a tautology.

The pipeline therefore restricts to the **32 architectures where S=0**, where
`nl_fraction` varies freely (mean ≈ 0.73, std ≈ 0.21) and can act as a genuine
mediator between design decisions and cryptographic performance.

This gives **960 observations per fitness** (32 architectures × 30 runs), which
is sufficient for CPT estimation and conditional independence tests.

---

## Notebook 1 — SBN_BN_sampler_v0.ipynb

### Configuration (Step 4)

```python
SELECTED_FITNESS = "algebraic"   # "algebraic" | "linear" | "differential"
POPULATION_SIZE  = 10
NUM_GENERATIONS  = 30
MUTATION_RATE    = 2.0
N_RUNS           = 30            # independent replications per architecture — do not reduce
```

**Note on `differential` fitness**: at ~12s per evaluation, the sampler switches
automatically to random sampling (pop=1, gen=1) for `differential`. Scores reflect
raw architectural capacity, not GA optima. CPT variance is higher than for
`algebraic`/`linear` — this is expected and documented.

### Estimated runtimes (30 runs × 32 architectures)

| Fitness | Mode | Estimated time |
|---------|------|----------------|
| `algebraic` | GA (pop=10, gen=30) | ~2h15 |
| `linear` | GA (pop=10, gen=30) | ~2h15 |
| `differential` | random sampling (pop=1, gen=1) | ~3h12 |

Set `RUN = True` in the execution cell to start.

### Output

One CSV written to the **current working directory** (not `data/`) with naming:

```
multirep_{fitness}_pop{population}_gen{generations}_n{runs}.csv
```

Example: `multirep_algebraic_pop10_gen30_n30.csv`

**Move this file to `data/` before running the analysis notebook.**

### Output schema

```
run_id | arch_id | Architecture | S | A | R | I | H | L | seed
      | best_score | nl_fraction | dep_density | avg_depth
      | confusion | diffusion | time_s
```

The five additional metrics beyond `best_score` are:

| Metric | Layer | Description |
|--------|-------|-------------|
| `nl_fraction` | 2 — Topological | Fraction of nonlinear gates (AND, OR) |
| `dep_density` | 2 — Topological | Density of the 16×16 input→output dependency matrix |
| `avg_depth` | 2 — Topological | Mean topological depth of the 16 output nodes |
| `confusion` | 3 — Shannon | Normalised linear resistance ∈ [0, 1] (1 = perfect confusion) |
| `diffusion` | 3 — Shannon | Average avalanche coefficient — SAC approximation (0.5 = perfect diffusion) |

### Step 5 — Convergence checks

Two mandatory checks before proceeding to the analysis:

- **Check A** — multi-run median vs reference GA best score. Decision rule: if more
  than 20% of architectures show a multi-run median below 90% of the reference,
  increase `NUM_GENERATIONS` and rerun.
- **Check B** — score variance across 30 runs. Architectures with zero variance
  produce deterministic CPT entries — acceptable and expected for highly constrained
  architectures.

---

## Notebook 2 — SBN_BN_analysis_v0.ipynb

### Input

Place all three `multirep_*.csv` files produced by the sampler in `data/` before
running:

```
data/
  multirep_algebraic_*.csv
  multirep_linear_*.csv
  multirep_differential_*.csv
```

The notebook loads the most recent file per fitness automatically.

### 4-layer causal architecture

```
Layer 1  Design decisions (S excluded — exogenous)
         A   R   I   H   L
                  │
                  ▼
Layer 2  Topological properties
         nl_fraction_class   dep_density_class   avg_depth_class
                  │
                  ▼
Layer 3  Shannon mechanisms
         confusion_class          diffusion_class
                  │
                  ▼
Layer 4  Cryptographic performance
         Fit_top_algebraic   Fit_top_linear   Fit_top_differential
```

Layer 2 and 3 variables are discretised into terciles (low / mid / high).
Layer 4 variables are binary (median split per fitness).

### Notebook walkthrough

| Step | Content |
|------|---------|
| **0** | Imports, data loading, S=0 restriction, node definitions |
| **1** | Convergence check on the S=0 subsample |
| **2** | Aggregation and discretisation — 32 architecture profiles, CPT helpers (MLE + Laplace smoothing), BIC scoring |
| **3** | FCA on two causally ordered contexts: **Context A** (design → mechanisms) and **Context B** (mechanisms → fitness). Stem Base rules guide BN parent selection |
| **4** | 4-layer BN: constrained structure learning (no backward edges between layers), CPT estimation, DAG visualisation |
| **5** | Shannon mediation test — partial correlation analysis to detect bypass edges from L1/L2 directly to L4 |
| **6** | Inference queries: P(high fitness \| constraints), multi-objective constraint ranking, causal path decomposition (total = direct + via confusion + via diffusion) |

### Key outputs

**Step 3 — FCA rules (two contexts)**

Context A reveals which constraints produce which topological and Shannon
properties. Context B reveals which mechanisms are necessary for high performance.
Fitness nodes never appear as premises — causal order is enforced.

**Step 5 — Shannon mediation test**

Tests whether confusion and diffusion fully mediate the effect of design constraints
on fitness. A bypass edge from L1 or L2 to L4 indicates a structural effect that
Shannon's mechanisms do not capture.

**Step 6 — Inference queries**

- *Query 1*: P(Fit_top | constraint combination) per fitness
- *Query 2*: multi-objective ranking — worst-case P(Fit_top) across all three fitness functions
- *Query 3*: causal path decomposition per constraint — direct effect vs indirect via confusion vs indirect via diffusion

---

## Relationship to fca/

| `fca/SBN_analysis_v0.ipynb` | `bn_analysis/` |
|---|---|
| Which constraint *combinations* co-occur with high fitness | *Why*: which causal mechanisms mediate the effect |
| 64 architectures, single GA run per architecture | 32 architectures (S=0), 30 GA runs per architecture |
| FCA Stem Base rules with support and coverage | 4-layer BN with CPT estimation and probabilistic inference |

The two pipelines are complementary. FCA gives exact combinatorial rules;
the BN gives probabilistic causal estimates with uncertainty.

---

## Limits

- N=32 architectures (S=0 only): CPTs for Layer 2/3 nodes are estimated on small sub-groups.
- `differential` mode uses random sampling (pop=1), not GA optima: CPT variance is higher.
- Tercile discretisation of Layer 2/3 loses quantitative information.
- Results reflect observational associations, not interventional causal effects.
