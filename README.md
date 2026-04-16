# sbn-explorer

Exploration of **Synchronous Boolean Networks** (SBNs) over a structured design space of 64 architectures defined by 6 orthogonal binary constraints. The project combines a genetic algorithm (GA) optimizer, a Formal Concept Analysis (FCA) pipeline, and a Bayesian Network (BN) analysis pipeline.

---

## Design Space

Each architecture is a binary vector in {0,1}⁶. All 64 combinations are fully independent and constructive — constraints are built into the network generator, not filtered post-hoc.

| Symbol | Constraint | Definition |
|--------|------------|------------|
| **S** | Stratification | Alternating nonlinear / linear layers |
| **A** | Acyclicity | Strict DAG — no feedback cycles |
| **R** | Regularity | Equal source-to-sink path lengths |
| **I** | Interleaving | Cross-block dependencies (B0 ↔ B1) |
| **H** | Homogeneity | Identical operations within each layer |
| **L** | Locality | Bounded connection distance |

**Circuit spec**: 16-bit input · 16-bit output · 16-bit state

---

## Repository Structure

```
sbn-explorer/
│
├── core/                        # Part 1 — SBN generator & genetic algorithm
│   ├── sbn_module_v0.py         # Core library (SBN class, GA, fitness functions)
│   └── SBN_generator_v0.ipynb  # Interactive GA runner for all 64 architectures
│
├── fca/                         # Part 2 — Formal Concept Analysis (depends on Part 1)
│   └── SBN_analysis_v0.ipynb
│
├── bn_analysis/                 # Part 3 — Bayesian Network analysis (depends on Part 1)
│   ├── SBN_BN_sampler_v0.ipynb
│   └── SBN_BN_analysis_v0.ipynb
│
└── data/                        # CSV results produced by Part 1 (git-ignored)
    └── README.md                # CSV format specification
```

**Dependency graph**

```
core/  ──┬──▶  fca/
         └──▶  bn_analysis/
```

Parts 2 and 3 each depend on Part 1 but are fully independent of each other.

---

## Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python | ≥ 3.10 | tested on 3.13 |
| numpy | any recent | |
| pandas | any recent | |
| matplotlib | any recent | |
| cupy | optional | GPU acceleration for `linear` and `algebraic` fitness |

Install core dependencies:

```bash
pip install numpy pandas matplotlib
```

Install GPU support (optional but strongly recommended for `differential` fitness):

```bash
pip install cupy-cuda12x   # adjust suffix to your CUDA version
```

The module detects CuPy at import time and falls back to CPU automatically if it is not available.

---

## Quickstart

### Step 1 — Run the genetic algorithm (Part 1)

Open `core/SBN_generator_v0.ipynb` and run cells in order.

**Key parameters to configure in Step 4a:**

```python
SELECTED_FITNESS    = "linear"   # "linear" | "algebraic" | "differential"
SELECTED_CONSTRAINT = "ALL"      # "ALL" | None | "S" | "A" | "R" | "I" | "H" | "L"
POPULATION_SIZE     = 20
NUM_GENERATIONS     = 50
MUTATION_RATE       = 3.0
```

Then set `RUN_GA = True` in Step 4b and run the cell.

**Estimated runtimes for all 64 architectures (pop=20, gen=50):**

| Fitness | Time per eval | Total (64 archs) |
|---------|--------------|-----------------|
| `linear` | ~0.01 s (GPU) | ~10 minutes |
| `algebraic` | ~0.01 s (GPU) | ~10 minutes |
| `differential` | ~12 s (GPU) | ~8.6 days |

**Output:** at the end of Step 4b, a CSV file is written to `../data/` with the following naming convention:

```
data/ga_results_{fitness}_pop{population}_gen{generations}.csv
```

Example: `data/ga_results_linear_pop20_gen50.csv`

**CSV columns:** `Rank`, `Architecture`, `S`, `A`, `R`, `I`, `H`, `L`, `Best_Score`, `Time_s`

The file must contain exactly **64 rows** (one per architecture) for downstream notebooks to work.

---

### Step 2 — FCA analysis (Part 2)

Open `fca/SBN_analysis_v0.ipynb`.

At the top of the **Shared Setup** cell, set `CSV_PATH` to the CSV produced in Step 1:

```python
CSV_PATH = '../data/ga_results_linear_pop20_gen50.csv'
```

Run all cells in order. The notebook performs:

- Completeness check and multi-level score discretisation
- Hypercube edge analysis
- Epistasis matrix
- FCA lattice construction and minimal rule extraction
- Post-lattice validation

---

### Step 3 — Bayesian Network analysis (Part 3)

Open `bn_analysis/SBN_BN_sampler_v0.ipynb` first. Configure `CSV_PATH` to the same CSV:

```python
CSV_PATH = '../data/ga_results_linear_pop20_gen50.csv'
```

Run all cells. The sampler generates one BN sample per architecture and writes its output to `data/`.

Then open `bn_analysis/SBN_BN_analysis_v0.ipynb`, configure its input path to the sampler output, and run all cells.

---

## Fitness Functions

| Name | Symbol | Measures | Direction |
|------|--------|----------|-----------|
| Linear resistance | `linear` | Walsh–Hadamard nonlinearity (max = 32640) | maximize |
| Algebraic degree | `algebraic` | ANF degree | maximize |
| Differential resistance | `differential` | DDT-based uniformity | minimize |

Parts 2 and 3 can be applied to results from **any** of the three fitness functions. Run Part 1 separately for each fitness you want to analyse.

---

## Reproducibility

CSV files are not versioned in this repository. All results can be reproduced by running `core/SBN_generator_v0.ipynb` with the GA parameters of your choice. The random seeds are fixed in the notebook (`np.random.seed(42)`, `random.seed(42)`).
