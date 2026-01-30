# neogel

**Neuro-Evolutionary Optimization & Genetic Evolutionary Learning**

`neogel` is a research-oriented Python framework for evolutionary computation and quality-diversity (QD) methods.
It is designed for **rapid experimentation**, **reproducibility**, and **extensibility**, with a focus on evolutionary algorithms, MAP-Elites–style QD, and expensive simulation-based problems.

Unlike general-purpose libraries, `neogel` prioritizes:
- clear experiment structure
- explicit control over evaluation, logging, and parallelism
- easy customization for research workflows

This repository currently supports:
- Genetic Algorithms (GA)
- MAP-Elites (standard quality-diversity)
- Config-driven experiments (YAML)
- Parallel evaluation via multiprocessing
- Automatic logging and visualization artifacts

---

## Installation

```bash
git clone <repo-url>
cd neogel
pip install -e .
```

Optional (for BipedalWalker experiments):
```bash
pip install "gymnasium[box2d]"
```

---

## Quickstart

Run a simple GA experiment on the Sphere function:

```bash
neogel run configs/ga/sphere.yaml
```

Run MAP-Elites (QD) on a fast Sphere benchmark:

```bash
neogel run configs/qd/sphere_mapelites.yaml
```

Run MAP-Elites on BipedalWalker with parallel evaluation:

```bash
neogel run configs/qd/bipedalwalker_mapelites.yaml --workers 4
```

Each run produces a structured output directory:

```
runs/<experiment>/<timestamp>_seedX/
├── metrics.csv
├── metrics.jsonl
└── artifacts/
    ├── fitness_curve.png
    └── archive_heatmap.png
```

---

## Core Concepts

### Engine
An **engine** implements an evolutionary algorithm:
- `GAEngine`
- `MapElitesEngine`

Engines follow a standard `ask → evaluate → tell` interface and expose metrics for logging.

---

### Problem
A **problem** defines how a genotype is evaluated:

```python
EvalRecord(
  objectives=np.array([...]),
  extras={...}  # optional behavior statistics for QD
)
```

Problems are referenced by import path in YAML configs.

---

### Descriptor (Quality-Diversity)
For MAP-Elites, a **descriptor function** maps an evaluated individual to a low-dimensional behavior descriptor using `EvalRecord.extras`.

Descriptors define the archive axes.

---

### Logging & Artifacts
Logging is modular and configurable:
- CSV and JSONL metrics
- live terminal visualization (Rich)
- automatic plots (fitness curves, archive heatmaps)

---

## Parallel Evaluation

Enable multiprocessing with:

```bash
neogel run configs/qd/bipedalwalker_mapelites.yaml --workers 4
```

Parallel evaluation is optional and most useful for expensive fitness functions.

---

## Project Status

**Version:** v0.1.0

This release focuses on:
- correctness
- reproducibility
- standard EC and QD workflows

Future releases will extend:
- emitters and schedulers
- additional EC algorithms
- richer QD descriptors and analysis tools

---

## License

MIT