# core — SBN Generator & Genetic Algorithm

This folder contains the core of the project: the SBN library (`sbn_module_v0.py`) and the interactive GA runner (`SBN_generator_v0.ipynb`).

---

## Files

| File | Role |
|------|------|
| `sbn_module_v0.py` | Library: SBN class, constraint builders, GA engine, fitness functions |
| `SBN_generator_v0.ipynb` | Interactive notebook: configure, run and export GA results |

---

## Module overview (`sbn_module_v0.py`)

### SBN representation

A `GenericSBN` is a directed computational graph of `ComputeNode` objects. Each node applies one of five operations: `AND`, `OR`, `XOR`, `NOT`, `IDENTITY`. The circuit has 16 input bits, 16 state bits and 16 output bits. Evaluation is synchronous: all nodes update in one step.

Constraints are **constructive** — they are built into the generator, not enforced by post-hoc filtering. Inactive constraints are explicitly broken so that each of the 64 architectures is structurally distinct.

### Constraint builders

| Constraint | Effect when active |
|------------|-------------------|
| **S** — Stratification | Odd layers use nonlinear ops (`AND`, `OR`), even layers use linear ops (`XOR`, `NOT`) |
| **A** — Acyclicity | Strict DAG: no feedback edges, evaluation order is topological |
| **R** — Regularity | Each gate reads only from the immediately preceding layer (equal path lengths) |
| **I** — Interleaving | Layer-1 gates draw inputs from both bit blocks B0 (bits 0–7) and B1 (bits 8–15) |
| **H** — Homogeneity | All gates within a layer share a single operation |
| **L** — Locality | Connections are restricted to circular distance ≤ 4 on the 16-bit ring |

### Fitness functions

Three cryptographic quality metrics are available, all implemented in `RewardFunctions`:

| Name | Description | Acceleration | Direction |
|------|-------------|-------------|-----------|
| `linear_resistance` | Walsh–Hadamard nonlinearity (max = 32640) | GPU (CuPy) | maximize |
| `algebraic_degree` | Algebraic Normal Form degree | GPU (CuPy) | maximize |
| `differential_resistance` | Difference Distribution Table uniformity | CPU + GPU DDT | minimize |

GPU acceleration requires CuPy. The `GPUAccelerator` class detects it at import time and falls back to CPU automatically.

### Genetic algorithm

`run_genetic_algorithm()` implements elitist GA with Poisson-distributed mutation rate. Key properties:

- Elite selection: top 20% of population carried over each generation
- Mutations preserve active constraints and maintain inactive constraint violations
- Seeds: fixed at `np.random.seed(42)` and `random.seed(42)` in the notebook

---

## Notebook walkthrough (`SBN_generator_v0.ipynb`)

| Step | Content |
|------|---------|
| 0 | Install dependencies |
| 1 | Module setup and CUDA path configuration |
| 2 | Imports |
| 2b | GPU memory check and cleanup |
| 3a | GPU initialisation (`GPUAccelerator`) |
| 3b | Truth table benchmark (CPU vs GPU) |
| 3c | Differential fitness performance notes |
| 3d | Architecture validation (all 64 combinations) |
| 4a | **GA configuration** — edit parameters here |
| 4b | **Run GA** — set `RUN_GA = True` to execute |
| 5 | Results visualisation |
| 6 | Post-GA validation |

### Parameters (Step 4a)

```python
SELECTED_FITNESS    = "linear"   # "linear" | "algebraic" | "differential"
SELECTED_CONSTRAINT = "ALL"      # "ALL" | None | "S" | "A" | "R" | "I" | "H" | "L"
POPULATION_SIZE     = 20
NUM_GENERATIONS     = 50
MUTATION_RATE       = 3.0        # mean number of mutations per child (Poisson)
```

Set `SELECTED_CONSTRAINT = "ALL"` to run the GA across all 64 architectures in a single execution.

### Output

Results are written to `../data/` with the naming convention:

```
../data/ga_results_{fitness}_pop{population}_gen{generations}.csv
```

Example: `../data/ga_results_linear_pop20_gen50.csv`

The CSV contains exactly 64 rows (one per architecture) with columns:
`Rank`, `Architecture`, `S`, `A`, `R`, `I`, `H`, `L`, `Best_Score`, `Time_s`

This file is the input expected by both `fca/` and `bn_analysis/`.

---

## Estimated runtimes (pop=20, gen=50, all 64 architectures)

| Fitness | Time per eval | Total |
|---------|--------------|-------|
| `linear` | ~0.01 s (GPU) | ~10 min |
| `algebraic` | ~0.01 s (GPU) | ~10 min |
| `differential` | ~12 s (GPU) | ~8.6 days |
