#!/usr/bin/env python3
"""Publication-oriented BBOB benchmark for operator-guided SOMA variants.

The benchmark uses the official COCO/BBOB noiseless suite and evaluates all
functions f1--f24 by default. Evaluation budgets are specified as evaluations
per dimension (FE/D), so each absolute budget is ``dimension * FE_per_D``.

Default algorithms
------------------
- Canonical-SOMA: classical All-to-One SOMA baseline with literature-style
  PRT=0.1, Step=0.11, and PathLength=3.0.
- GC-SOMA: geometry-controlled SOMA.
- RA-SOMA: covariance-eigenbasis rotation-aware SOMA.
- iSOMA: the supplied ISOMAOptimizer loaded at runtime.
- iL-SHOMA-RA: experimental iSOMA extension combining rotation-aware masking,
  success-history adaptation, and linear population-size reduction.
- SciPy-DE: scipy.optimize.differential_evolution, best1bin.
- iL-SHADE: PyADE iL-SHADE with its package-default population size.
- iL-SHADE-matched: PyADE iL-SHADE with initial population 4D by default.

Optional RA ablations
---------------------
- RA-SOMA-random: a fresh random orthogonal masking basis each migration.
- RA-SOMA-fixed: one random orthogonal masking basis fixed for the run.

Main outputs retain the filenames of the previous benchmark:
``benchmark_raw.csv``, ``convergence_curves.csv``, ``target_hits_raw.csv``,
``final_performance_summary.csv``, ``function_instance_block_scores.csv``,
``average_ranks.csv``, ``target_attainment_by_function.csv``,
``target_attainment_overall.csv``, ``friedman_tests.csv``,
``pairwise_wilcoxon_holm.csv``, ``overall_algorithm_summary.csv``, and
``PUBLICATION_SUMMARY.md``. Additional appendix-oriented outputs include
``mechanism_diagnostics.csv``, target-ECDF plots, average-rank plots, and
geometry-control diagnostic plots.

Example
-------
    python -m pip install coco-experiment numpy pandas matplotlib scipy pyade
    python bbob_full_soma_benchmark.py \
        --dimensions 5,10,20 \
        --budgets-per-dimension 500,2500 \
        --instances 1-5 \
        --seeds 10 \
        --isoma-file iSOMA.py \
        --outdir bbob_full_results

        python bbob_full_soma_benchmark.py \
                --dimensions 5 \
                --budgets-per-dimension 50 \
                --instances 1-5 \
                --seeds 10 \
                --isoma-file iSOMA.py \
                --outdir bbob_full_results

Notes
-----
- The supplied iSOMA file is loaded without modification.
- PyADE iL-SHADE is strict by default. If PyADE cannot be imported, the script
  raises an error rather than silently substituting another algorithm. Use
  ``--allow-ilshade-fallback`` only for development checks; fallback results
  are explicitly labelled and should not be reported as iL-SHADE.
- Absolute objective errors are the primary final-performance measure. Logarithmic
  columns are retained only as secondary diagnostics and for log-scale figures.
- Repeated seeds are aggregated within function-instance blocks before ranks and
  paired statistical tests are computed.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution

ERROR_FLOOR = 1e-300
PLOT_ERROR_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Complete noiseless BBOB function groups
# ---------------------------------------------------------------------------
BBOB_GROUPS: dict[str, dict[str, Any]] = {
    "g1": {
        "number": 1,
        "name": "Separable functions",
        "slug": "group1_separable",
        "functions": {
            1: "Sphere Function",
            2: "Separable Ellipsoidal Function",
            3: "Separable Rastrigin Function",
            4: "Bueche--Rastrigin Function",
            5: "Linear Slope Function",
        },
    },
    "g2": {
        "number": 2,
        "name": "Low or moderate conditioning",
        "slug": "group2_low_moderate_conditioning",
        "functions": {
            6: "Attractive Sector Function",
            7: "Step Ellipsoidal Function",
            8: "Rosenbrock Function, original",
            9: "Rosenbrock Function, rotated",
        },
    },
    "g3": {
        "number": 3,
        "name": "High conditioning and unimodal",
        "slug": "group3_high_conditioning_unimodal",
        "functions": {
            10: "Ellipsoidal Function",
            11: "Discus Function",
            12: "Bent Cigar Function",
            13: "Sharp Ridge Function",
            14: "Different Powers Function",
        },
    },
    "g4": {
        "number": 4,
        "name": "Multimodal with adequate global structure",
        "slug": "group4_multimodal_adequate_structure",
        "functions": {
            15: "Rastrigin Function",
            16: "Weierstrass Function",
            17: "Schaffer's F7 Function",
            18: "Schaffer's F7 Function, moderately ill-conditioned",
            19: "Composite Griewank--Rosenbrock Function F8F2",
        },
    },
    "g5": {
        "number": 5,
        "name": "Multimodal with weak global structure",
        "slug": "group5_multimodal_weak_structure",
        "functions": {
            20: "Schwefel Function",
            21: "Gallagher's Gaussian 101-me Peaks Function",
            22: "Gallagher's Gaussian 21-hi Peaks Function",
            23: "Katsuura Function",
            24: "Lunacek bi-Rastrigin Function",
        },
    },
}

FUNCTION_TO_GROUP: dict[int, str] = {
    function_id: group_id
    for group_id, group in BBOB_GROUPS.items()
    for function_id in group["functions"]
}
ALL_FUNCTION_IDS = sorted(FUNCTION_TO_GROUP)

ALGORITHM_SPECS: dict[str, tuple[str, Optional[str]]] = {
    "Canonical-SOMA": ("canonical_soma", None),
    "GC-SOMA": ("operator_soma", "geometry"),
    "RA-SOMA": ("operator_soma", "rotation"),
    "RA-SOMA-random": ("operator_soma", "random_rotation"),
    "RA-SOMA-fixed": ("operator_soma", "fixed_rotation"),
    "iSOMA": ("isoma", None),
    "iL-SHOMA-RA": ("ilshoma_ra", None),
    "SciPy-DE": ("scipy_de", None),
    "iL-SHADE": ("ilshade", "matched")
}

DEFAULT_ALGORITHMS = [
    "Canonical-SOMA",
    "GC-SOMA",
    "RA-SOMA",
    "iSOMA",
    "iL-SHOMA-RA",
    "SciPy-DE",
    "iL-SHADE"
]

PLOT_STYLES: dict[str, dict[str, Any]] = {
    "Canonical-SOMA": {"color": "#000000", "linestyle": "-", "marker": "o"},
    "GC-SOMA": {"color": "#E69F00", "linestyle": "--", "marker": "s"},
    "RA-SOMA": {"color": "#56B4E9", "linestyle": "-.", "marker": "^"},
    "RA-SOMA-random": {"color": "#CC79A7", "linestyle": ":", "marker": "h"},
    "RA-SOMA-fixed": {"color": "#999999", "linestyle": "--", "marker": "<"},
    "iSOMA": {"color": "#D55E00", "linestyle": "-.", "marker": "*"},
    "iL-SHOMA-RA": {"color": "#009E73", "linestyle": "-", "marker": "D"},
    "SciPy-DE": {"color": "#F0E442", "linestyle": "-", "marker": "v"},
    "iL-SHADE": {"color": "#0072B2", "linestyle": "--", "marker": "P"},
    "iL-SHADE-matched": {"color": "#332288", "linestyle": ":", "marker": "X"},
    "iL-SHADE-fallback": {"color": "#0072B2", "linestyle": ":", "marker": "x"},
    "iL-SHADE-matched-fallback": {"color": "#332288", "linestyle": "-.", "marker": "+"},
}


# ---------------------------------------------------------------------------
# Parsing and utility helpers
# ---------------------------------------------------------------------------
def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value not in values:
                values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            value = float(token)
            if value <= 0:
                raise argparse.ArgumentTypeError("Values must be positive.")
            if value not in values:
                values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one positive number.")
    return values


def parse_name_list(text: str) -> list[str]:
    values = [token.strip() for token in text.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one name.")
    return values


def parse_index_spec(text: str) -> list[int]:
    """Parse specifications such as ``1-5,8,10-12``."""
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if match:
            start, stop = int(match.group(1)), int(match.group(2))
            if stop < start:
                raise argparse.ArgumentTypeError(f"Invalid range: {token}")
            values.update(range(start, stop + 1))
        else:
            values.add(int(token))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("Indices must be positive integers.")
    return sorted(values)


def stable_seed(*components: int) -> int:
    sequence = np.random.SeedSequence([int(v) & 0xFFFFFFFF for v in components])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def error_to_log10(error: float) -> float:
    return math.log10(max(float(error), ERROR_FLOOR))


def pad_history(history: np.ndarray, max_evals: int) -> np.ndarray:
    history = np.asarray(history, dtype=float)
    if max_evals <= 0:
        return np.empty(0, dtype=float)
    if history.size == 0:
        return np.full(max_evals, np.inf, dtype=float)
    if history.size >= max_evals:
        return history[:max_evals]
    return np.pad(history, (0, max_evals - history.size), constant_values=history[-1])


def sample_history(
    history: np.ndarray,
    max_evals: int,
    points: int = 201,
) -> tuple[np.ndarray, np.ndarray]:
    linear = np.linspace(1, max_evals, points, dtype=int)
    geometric = np.geomspace(1, max_evals, points).astype(int)
    grid = np.unique(np.concatenate(([1, max_evals], linear, geometric)))
    padded = pad_history(history, max_evals)
    return grid, padded[grid - 1]


def first_target_hit(history: np.ndarray, target: float) -> Optional[int]:
    hits = np.flatnonzero(np.asarray(history, dtype=float) <= float(target))
    return int(hits[0] + 1) if hits.size else None


def normalized_log_auc(history: np.ndarray, max_evals: int) -> float:
    padded = pad_history(history, max_evals)
    return float(np.mean(np.log10(np.maximum(padded, ERROR_FLOOR))))


def normalized_absolute_auc(history: np.ndarray, max_evals: int) -> float:
    """Mean best-so-far error normalized by the first finite error.

    This is bounded to [0, 1] for monotone best-so-far histories and is more
    comparable across heterogeneous BBOB functions than raw absolute AUC.
    """
    padded = pad_history(history, max_evals)
    finite = padded[np.isfinite(padded)]
    if finite.size == 0:
        return np.inf
    scale = max(float(finite[0]), ERROR_FLOOR)
    return float(np.mean(np.minimum(padded / scale, 1.0)))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return []
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(p)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def random_orthogonal_basis(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def covariance_basis(population: np.ndarray) -> np.ndarray:
    """Regularized eigenbasis of a population stored as N x D."""
    dimension = population.shape[1]
    if population.shape[0] < 2:
        return np.eye(dimension)
    centered = population - np.mean(population, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, population.shape[0])
    trace = float(np.trace(covariance))
    regularizer = 1e-12 * (trace / dimension if trace > 0 else 1.0)
    covariance = covariance + regularizer * np.eye(dimension)
    _, basis = np.linalg.eigh(covariance)
    return basis


def make_legacy_initial_population(
    seed: int,
    size: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Match ``np.random.rand(D, N)`` used by the supplied iSOMA code."""
    state = np.random.RandomState(int(seed))
    matrix = lower[:, None] + state.rand(len(lower), int(size)) * (upper - lower)[:, None]
    return matrix.T.copy()


# ---------------------------------------------------------------------------
# Official COCO/BBOB problem adapter and exact FE accounting
# ---------------------------------------------------------------------------
def _bbob2009_uniform(count: int, seed: int) -> np.ndarray:
    state = float(abs(int(seed)))
    if state < 1.0:
        state = 1.0
    shuffle = np.zeros(32, dtype=float)
    for i in range(39, -1, -1):
        quotient = math.floor(state / 127773.0)
        state = 16807.0 * (state - quotient * 127773.0) - 2836.0 * quotient
        if state < 0.0:
            state += 2147483647.0
        if i < 32:
            shuffle[i] = state
    current = float(shuffle[0])
    out = np.empty(int(count), dtype=float)
    for i in range(int(count)):
        quotient = math.floor(state / 127773.0)
        state = 16807.0 * (state - quotient * 127773.0) - 2836.0 * quotient
        if state < 0.0:
            state += 2147483647.0
        index = int(math.floor(current / 67108865.0))
        current = float(shuffle[index])
        shuffle[index] = state
        out[i] = current / 2.147483647e9
    out[out == 0.0] = 1e-99
    return out


def _bbob2009_gauss(count: int, seed: int) -> np.ndarray:
    u = _bbob2009_uniform(2 * int(count), seed)
    u1 = np.maximum(u[:count], np.finfo(float).tiny)
    u2 = u[count:]
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def bbob2009_compute_fopt(function_id: int, instance_id: int) -> float:
    seed = int(function_id) + 10000 * int(instance_id)
    numerator = float(_bbob2009_gauss(1, seed)[0])
    denominator = float(_bbob2009_gauss(1, seed + 1)[0])
    ratio = math.copysign(np.inf, numerator) if denominator == 0.0 else numerator / denominator
    value = np.round(10000.0 * ratio) / 100.0
    return float(np.clip(value, -1000.0, 1000.0))


@dataclass(frozen=True)
class BBOBProblemAdapter:
    coco_problem: Any
    function_id: int
    function_name: str
    group_id: str
    group_name: str
    instance: int
    dim: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    initial_solution: np.ndarray
    fopt: float
    objective_reference: str

    @classmethod
    def from_coco(cls, problem: Any) -> "BBOBProblemAdapter":
        function_id = int(problem.id_function)
        instance = int(problem.id_instance)
        group_id = FUNCTION_TO_GROUP[function_id]
        group = BBOB_GROUPS[group_id]
        return cls(
            coco_problem=problem,
            function_id=function_id,
            function_name=str(group["functions"][function_id]),
            group_id=group_id,
            group_name=str(group["name"]),
            instance=instance,
            dim=int(problem.dimension),
            lower_bounds=np.asarray(problem.lower_bounds, dtype=float).copy(),
            upper_bounds=np.asarray(problem.upper_bounds, dtype=float).copy(),
            initial_solution=np.asarray(problem.initial_solution, dtype=float).copy(),
            fopt=bbob2009_compute_fopt(function_id, instance),
            objective_reference="COCO BBOB f(x)-fopt",
        )

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return list(zip(self.lower_bounds.tolist(), self.upper_bounds.tolist()))

    @property
    def label(self) -> str:
        return f"f{self.function_id:02d} {self.function_name}"

    def __call__(self, x: np.ndarray) -> float:
        raw = float(self.coco_problem(np.asarray(x, dtype=float)))
        return max(raw - self.fopt, 0.0)


class BudgetedObjective:
    """Exact true-evaluation accounting with best-so-far history."""

    def __init__(self, func: Callable[[np.ndarray], float], max_evals: int):
        self.func = func
        self.max_evals = int(max_evals)
        self.eval_count = 0
        self.attempted_calls = 0
        self.best_raw = np.inf
        self.best_error = np.inf
        self.best_x: Optional[np.ndarray] = None
        self.history_true: list[float] = []
        self.history_noisy: list[float] = []

    def __call__(self, x: np.ndarray) -> float:
        self.attempted_calls += 1
        if self.eval_count >= self.max_evals:
            return np.inf
        x = np.asarray(x, dtype=float)
        value = float(self.func(x))
        self.eval_count += 1
        if value < self.best_raw:
            self.best_raw = value
            self.best_error = max(value, 0.0)
            self.best_x = x.copy()
        self.history_true.append(self.best_error)
        self.history_noisy.append(self.best_error)
        return value


@dataclass
class OptimizerRun:
    best_x: np.ndarray
    best_error: float
    evals: int
    history: np.ndarray
    diagnostics: dict[str, Any]
    mechanism_rows: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Boundary repair
# ---------------------------------------------------------------------------
def repair_clip(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(x, lower, upper)


def repair_reflect(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = upper - lower
    if np.any(span <= 0):
        raise ValueError("Upper bounds must exceed lower bounds.")
    y = np.mod(x - lower, 2.0 * span)
    y = np.where(y <= span, y, 2.0 * span - y)
    return lower + y


def repair_midpoint(
    x: np.ndarray,
    parent: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    below = y < lower
    above = y > upper
    y[below] = 0.5 * (parent[below] + lower[below])
    y[above] = 0.5 * (parent[above] + upper[above])
    return y


def apply_repair(
    name: str,
    x: np.ndarray,
    parent: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if name == "clip":
        return repair_clip(x, lower, upper)
    if name == "reflect":
        return repair_reflect(x, lower, upper)
    if name == "midpoint":
        return repair_midpoint(x, parent, lower, upper)
    raise ValueError(f"Unknown repair method: {name}")


# ---------------------------------------------------------------------------
# Canonical SOMA baseline
# ---------------------------------------------------------------------------
@dataclass
class CanonicalSomaConfig:
    prt: float = 0.10
    step: float = 0.11
    path_length: float = 3.0
    repair: str = "clip"
    force_nonzero_mask: bool = False


def sample_mask(
    rng: np.random.Generator,
    dimension: int,
    probability: float,
    force_nonzero: bool = False,
) -> np.ndarray:
    mask = (rng.random(dimension) < probability).astype(float)
    if force_nonzero and not np.any(mask):
        mask[int(rng.integers(dimension))] = 1.0
    return mask


def run_canonical_soma(
    problem: BBOBProblemAdapter,
    max_evals: int,
    seed: int,
    initial_population: np.ndarray,
    cfg: CanonicalSomaConfig,
) -> OptimizerRun:
    """Canonical All-to-One SOMA with a common migration origin per path."""
    rng = np.random.default_rng(seed)
    population = np.asarray(initial_population, dtype=float).copy()
    lower, upper = problem.lower_bounds, problem.upper_bounds
    objective = BudgetedObjective(problem, max_evals)
    fitness = np.array([objective(x) for x in population], dtype=float)
    path_values = np.arange(cfg.step, cfg.path_length + 0.5 * cfg.step, cfg.step)
    migrations = 0
    proposals = 0
    successful_migrants = 0
    migrant_attempts = 0

    while objective.eval_count < max_evals:
        leader_idx = int(np.argmin(fitness))
        leader = population[leader_idx].copy()
        any_evaluated = False
        migrations += 1
        for i in range(len(population)):
            if i == leader_idx or objective.eval_count >= max_evals:
                continue
            migrant_attempts += 1
            origin = population[i].copy()
            origin_fit = float(fitness[i])
            best_x, best_fit = origin.copy(), origin_fit
            displacement = leader - origin
            for t in path_values:
                if objective.eval_count >= max_evals:
                    break
                mask = sample_mask(rng, problem.dim, cfg.prt, cfg.force_nonzero_mask)
                raw = origin + float(t) * mask * displacement
                trial = apply_repair(cfg.repair, raw, origin, lower, upper)
                value = objective(trial)
                proposals += 1
                any_evaluated = True
                if value < best_fit:
                    best_fit, best_x = value, trial.copy()
            if best_fit < origin_fit:
                population[i], fitness[i] = best_x, best_fit
                successful_migrants += 1
        if not any_evaluated:
            break

    if objective.best_x is None:
        raise RuntimeError("Canonical SOMA performed no valid evaluation.")
    return OptimizerRun(
        best_x=objective.best_x,
        best_error=float(objective.best_error),
        evals=objective.eval_count,
        history=np.asarray(objective.history_true, dtype=float),
        diagnostics={
            "backend": "native canonical SOMA All-to-One",
            "attempted_calls": objective.attempted_calls,
            "migrations": migrations,
            "proposals": proposals,
            "migrant_attempts": migrant_attempts,
            "successful_migrants": successful_migrants,
            "migrant_success_rate": successful_migrants / max(1, migrant_attempts),
            "prt": cfg.prt,
            "step": cfg.step,
            "path_length": cfg.path_length,
            "repair": cfg.repair,
            "force_nonzero_mask": cfg.force_nonzero_mask,
        },
    )


# ---------------------------------------------------------------------------
# Geometry-controlled and rotation-aware SOMA
# ---------------------------------------------------------------------------
@dataclass
class OperatorSomaConfig:
    n_path_points: int = 6
    static_prt: float = 0.30
    path_min: float = 0.40
    path_max: float = 2.40
    repair: str = "reflect"
    force_nonzero_mask: bool = False
    geometry_bins: int = 20


def geometry_targets(progress: float, dimension: int) -> tuple[float, float, float]:
    s = float(np.clip(progress, 0.0, 1.0))
    p = 0.35 + 0.55 * s
    rho = 0.85 - 0.70 * s
    rho = max(rho, 1.0 - p + 1e-6)
    return dimension * p, rho, p


def t_from_p_rho(p: float, rho: float, overshoot: bool) -> float:
    p = float(np.clip(p, 1e-12, 1.0))
    rho = float(np.clip(rho, 1.0 - p, 1.0))
    root = math.sqrt(max(0.0, 1.0 - (1.0 - rho) / p))
    return 1.0 + root if overshoot else 1.0 - root


def operator_path_parameters(
    variant: str,
    progress: float,
    dimension: int,
    cfg: OperatorSomaConfig,
) -> list[tuple[float, float, float]]:
    if variant in {"rotation", "random_rotation", "fixed_rotation"}:
        return [
            (cfg.static_prt, float(t), np.nan)
            for t in np.linspace(cfg.path_min, cfg.path_max, cfg.n_path_points)
        ]
    if variant == "geometry":
        _, rho_center, p = geometry_targets(progress, dimension)
        offsets = np.linspace(0.12, -0.12, cfg.n_path_points)
        params: list[tuple[float, float, float]] = []
        for offset in offsets:
            rho = float(np.clip(rho_center + offset, max(1.0 - p + 1e-6, 0.02), 0.98))
            t = t_from_p_rho(p, rho, overshoot=(progress < 0.55))
            params.append((p, t, rho))
        return params
    raise ValueError(f"Unsupported operator SOMA variant: {variant}")


def run_operator_soma(
    problem: BBOBProblemAdapter,
    max_evals: int,
    seed: int,
    initial_population: np.ndarray,
    variant: str,
    cfg: OperatorSomaConfig,
) -> OptimizerRun:
    rng = np.random.default_rng(seed)
    population = np.asarray(initial_population, dtype=float).copy()
    lower, upper = problem.lower_bounds, problem.upper_bounds
    objective = BudgetedObjective(problem, max_evals)
    fitness = np.array([objective(x) for x in population], dtype=float)
    fixed_basis = random_orthogonal_basis(rng, problem.dim) if variant == "fixed_rotation" else None

    successful_migrants = 0
    migrant_attempts = 0
    proposals = 0
    migrations = 0
    bins: list[dict[str, float]] = [
        {
            "count": 0.0,
            "target_k_sum": 0.0,
            "active_k_sum": 0.0,
            "target_rho_sum": 0.0,
            "empirical_rho_sum": 0.0,
            "rho_count": 0.0,
        }
        for _ in range(cfg.geometry_bins)
    ]

    while objective.eval_count < max_evals:
        leader_idx = int(np.argmin(fitness))
        leader = population[leader_idx].copy()
        if variant == "rotation":
            basis = covariance_basis(population)
        elif variant == "random_rotation":
            basis = random_orthogonal_basis(rng, problem.dim)
        elif variant == "fixed_rotation":
            basis = np.asarray(fixed_basis)
        else:
            basis = np.eye(problem.dim)
        any_evaluated = False
        migrations += 1

        for i in range(len(population)):
            if i == leader_idx or objective.eval_count >= max_evals:
                continue
            migrant_attempts += 1
            origin = population[i].copy()
            origin_fit = float(fitness[i])
            best_x, best_fit = origin.copy(), origin_fit
            progress = objective.eval_count / max_evals
            params = operator_path_parameters(variant, progress, problem.dim, cfg)
            displacement = leader - origin
            basis_displacement = basis.T @ displacement
            denominator = float(np.dot(origin - leader, origin - leader))

            for p, t, target_rho in params:
                if objective.eval_count >= max_evals:
                    break
                mask = sample_mask(rng, problem.dim, p, cfg.force_nonzero_mask)
                raw = origin + t * (basis @ (mask * basis_displacement))
                trial = apply_repair(cfg.repair, raw, origin, lower, upper)
                value = objective(trial)
                proposals += 1
                any_evaluated = True
                if variant == "geometry":
                    bin_index = min(cfg.geometry_bins - 1, int(progress * cfg.geometry_bins))
                    active_k = float(np.sum(mask))
                    empirical_rho = np.nan
                    if denominator > 0:
                        empirical_rho = float(np.dot(raw - leader, raw - leader) / denominator)
                    target_k, _, _ = geometry_targets(progress, problem.dim)
                    record = bins[bin_index]
                    record["count"] += 1.0
                    record["target_k_sum"] += target_k
                    record["active_k_sum"] += active_k
                    record["target_rho_sum"] += float(target_rho)
                    if np.isfinite(empirical_rho):
                        record["empirical_rho_sum"] += empirical_rho
                        record["rho_count"] += 1.0
                if value < best_fit:
                    best_fit, best_x = value, trial.copy()

            if best_fit < origin_fit:
                population[i], fitness[i] = best_x, best_fit
                successful_migrants += 1
        if not any_evaluated:
            break

    mechanism_rows: list[dict[str, Any]] = []
    if variant == "geometry":
        for bin_index, record in enumerate(bins):
            if record["count"] <= 0:
                continue
            mechanism_rows.append(
                {
                    "mechanism": "geometry_control",
                    "progress_bin": bin_index,
                    "progress_midpoint": (bin_index + 0.5) / cfg.geometry_bins,
                    "proposal_count": int(record["count"]),
                    "target_active_dimension": record["target_k_sum"] / record["count"],
                    "empirical_active_dimension": record["active_k_sum"] / record["count"],
                    "target_contraction": record["target_rho_sum"] / record["count"],
                    "empirical_raw_contraction": (
                        record["empirical_rho_sum"] / record["rho_count"]
                        if record["rho_count"] > 0
                        else np.nan
                    ),
                }
            )

    if objective.best_x is None:
        raise RuntimeError(f"{variant} SOMA performed no valid evaluation.")
    return OptimizerRun(
        best_x=objective.best_x,
        best_error=float(objective.best_error),
        evals=objective.eval_count,
        history=np.asarray(objective.history_true, dtype=float),
        diagnostics={
            "backend": "native operator-guided SOMA",
            "variant": variant,
            "attempted_calls": objective.attempted_calls,
            "migrations": migrations,
            "proposals": proposals,
            "migrant_attempts": migrant_attempts,
            "successful_migrants": successful_migrants,
            "migrant_success_rate": successful_migrants / max(1, migrant_attempts),
            "repair": cfg.repair,
        },
        mechanism_rows=mechanism_rows,
    )

# ---------------------------------------------------------------------------
# Supplied iSOMA loader and adapter
# ---------------------------------------------------------------------------
@dataclass
class ISomaConfig:
    n_jump: int = 10
    step: float = 0.3
    max_migration: int = 10000
    m: int = 10
    n: int = 5
    k: int = 15


def load_isoma_class(file_path: Path) -> type:
    file_path = file_path.expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(
            f"iSOMA file not found: {file_path}. Pass --isoma-file explicitly."
        )
    module_name = f"user_isoma_{abs(hash(str(file_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "qiskit_algorithms":
            raise ModuleNotFoundError(
                "The supplied iSOMA file requires qiskit-algorithms. Install it "
                "with `python -m pip install qiskit-algorithms`."
            ) from exc
        raise
    optimizer_class = getattr(module, "ISOMAOptimizer", None)
    if optimizer_class is None:
        raise AttributeError(f"{file_path} does not define ISOMAOptimizer.")
    return optimizer_class


def run_isoma(
    problem: BBOBProblemAdapter,
    max_evals: int,
    seed: int,
    population_size: int,
    optimizer_class: type,
    cfg: ISomaConfig,
) -> OptimizerRun:
    """Run the supplied implementation.

    The seed is also the common SOMA-family initialization seed. Because the
    supplied implementation initializes with ``np.random.rand(D, N)``, this
    matches ``make_legacy_initial_population`` for the same seed and size.
    """
    np.random.seed(seed)
    random.seed(seed)
    m = min(max(1, cfg.m), population_size)
    n = min(max(1, cfg.n), m)
    k = min(max(2, cfg.k), population_size)
    objective = BudgetedObjective(problem, max_evals)
    optimizer = optimizer_class(
        N_jump=int(cfg.n_jump),
        Step=float(cfg.step),
        PopSize=int(population_size),
        Max_Migration=int(cfg.max_migration),
        Max_FEs=int(max_evals),
        VarMin=float(np.min(problem.lower_bounds)),
        VarMax=float(np.max(problem.upper_bounds)),
        m=int(m),
        n=int(n),
        k=int(k),
        file_path=None,
        callback=None,
    )
    result = optimizer.minimize(
        objective,
        problem.initial_solution.copy(),
        bounds=problem.bounds,
    )
    if objective.best_x is None:
        best_x = np.asarray(result.x, dtype=float)
        best_error = max(float(result.fun), 0.0)
    else:
        best_x = objective.best_x
        best_error = objective.best_error
    return OptimizerRun(
        best_x=best_x,
        best_error=float(best_error),
        evals=objective.eval_count,
        history=np.asarray(objective.history_true, dtype=float),
        diagnostics={
            "backend": f"dynamic file: {optimizer_class.__module__}.ISOMAOptimizer",
            "attempted_calls": objective.attempted_calls,
            "reported_nfev": int(getattr(result, "nfev", -1)),
            "reported_nit": int(getattr(result, "nit", -1)),
            "effective_m": m,
            "effective_n": n,
            "effective_k": k,
            "initialization_seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Experimental iL-SHOMA-RA optimizer
# ---------------------------------------------------------------------------
@dataclass
class ILShomaRAConfig:
    n_jump: int = 10
    step: float = 0.3
    max_migration: int = 10000
    m: int = 10
    n: int = 5
    k: int = 15
    memory_size: int = 6
    prt_sigma: float = 0.10
    scale_cauchy: float = 0.10
    memory_blend: float = 0.70
    min_population_multiplier: float = 1.0
    minimum_population: int = 4
    repair: str = "clip"
    force_nonzero_mask: bool = False
    stagnation_factor: float = 50.0
    restart_fraction: float = 0.10
    max_path_scale: float = 2.0
    diagnostic_bins: int = 20


def _sample_positive_cauchy(
    rng: np.random.Generator,
    location: float,
    scale: float,
    lower: float,
    upper: float,
) -> float:
    for _ in range(100):
        value = float(location + scale * rng.standard_cauchy())
        if lower < value <= upper:
            return value
    return float(np.clip(location, lower + 1e-12, upper))


def _linear_population_target(
    initial_size: int,
    minimum_size: int,
    evaluations: int,
    max_evals: int,
) -> int:
    progress = float(np.clip(evaluations / max_evals, 0.0, 1.0))
    size = round(initial_size + (minimum_size - initial_size) * progress)
    return int(np.clip(size, minimum_size, initial_size))


def run_ilshoma_ra(
    problem: BBOBProblemAdapter,
    max_evals: int,
    seed: int,
    initial_population: np.ndarray,
    cfg: ILShomaRAConfig,
) -> OptimizerRun:
    """iSOMA + RA masking + success history + linear population reduction.

    The implementation preserves the main iSOMA structure: rank-selected
    migrants, tournament-selected leaders, decreasing path coefficients, and
    first-improvement acceptance. Success history adapts the PRT probability
    and a multiplicative scale applied to each original iSOMA path coefficient.
    This keeps the original decreasing path while allowing successful path
    scales to persist. Population size decreases linearly from N0 to at least D
    by default, which preserves a minimally informative covariance basis.
    """
    rng = np.random.default_rng(seed)
    population = np.asarray(initial_population, dtype=float).copy()
    lower, upper = problem.lower_bounds, problem.upper_bounds
    objective = BudgetedObjective(problem, max_evals)
    fitness = np.array([objective(x) for x in population], dtype=float)
    initial_size = len(population)
    minimum_size = max(
        int(cfg.minimum_population),
        int(math.ceil(cfg.min_population_multiplier * problem.dim)),
    )
    minimum_size = min(minimum_size, initial_size)

    memory_p = np.full(cfg.memory_size, 0.50, dtype=float)
    memory_scale = np.ones(cfg.memory_size, dtype=float)
    memory_cursor = 0
    migration = 0
    failed_migrant_count = 0
    best_cost_old = float(np.min(fitness))
    successful_migrants = 0
    migrant_attempts = 0
    proposal_count = 0
    restart_count = 0
    reduction_events = 0

    # Preserve the supplied iSOMA selection proportions as population changes.
    m_fraction = min(1.0, max(1, cfg.m) / initial_size)
    n_fraction_of_m = min(1.0, max(1, cfg.n) / max(1, cfg.m))
    k_fraction = min(1.0, max(2, cfg.k) / initial_size)

    bins: list[dict[str, float]] = [
        {
            "migrations": 0.0,
            "population_sum": 0.0,
            "selected_p_sum": 0.0,
            "selected_scale_sum": 0.0,
            "successes": 0.0,
            "attempts": 0.0,
            "memory_p_sum": 0.0,
            "memory_scale_sum": 0.0,
        }
        for _ in range(cfg.diagnostic_bins)
    ]

    while objective.eval_count < max_evals and migration < cfg.max_migration:
        migration += 1
        current_size = len(population)
        if current_size < 2:
            break
        progress = objective.eval_count / max_evals
        bin_index = min(cfg.diagnostic_bins - 1, int(progress * cfg.diagnostic_bins))
        bin_record = bins[bin_index]
        bin_record["migrations"] += 1.0
        bin_record["population_sum"] += current_size
        bin_record["memory_p_sum"] += float(np.mean(memory_p))
        bin_record["memory_scale_sum"] += float(np.mean(memory_scale))

        basis = covariance_basis(population)
        m_current = min(current_size, max(2, int(round(m_fraction * current_size))))
        n_current = min(m_current, max(1, int(round(n_fraction_of_m * m_current))))
        k_current = min(current_size, max(2, int(round(k_fraction * current_size))))

        migrant_pool = rng.choice(current_size, size=m_current, replace=False)
        migrant_order = migrant_pool[np.argsort(fitness[migrant_pool])]
        successful_parameters: list[tuple[float, float, float]] = []
        migration_global_improvement = False

        for migrant_idx in migrant_order[:n_current]:
            if objective.eval_count >= max_evals:
                break
            migrant_attempts += 1
            bin_record["attempts"] += 1.0
            origin = population[migrant_idx].copy()
            origin_fit = float(fitness[migrant_idx])

            leader_pool = rng.choice(current_size, size=k_current, replace=False)
            leader_order = leader_pool[np.argsort(fitness[leader_pool])]
            leader_idx = int(leader_order[0])
            if leader_idx == migrant_idx:
                if len(leader_order) < 2:
                    continue
                leader_idx = int(leader_order[1])
            leader = population[leader_idx].copy()
            displacement_basis = basis.T @ (leader - origin)

            accepted = False
            for move in range(1, cfg.n_jump + 1):
                if objective.eval_count >= max_evals:
                    break
                base_t = (cfg.n_jump - move + 1) * cfg.step
                memory_index = int(rng.integers(cfg.memory_size))
                baseline_p = 0.1 + 0.9 * (objective.eval_count / max_evals)
                p_center = (
                    cfg.memory_blend * memory_p[memory_index]
                    + (1.0 - cfg.memory_blend) * baseline_p
                )
                p = float(np.clip(rng.normal(p_center, cfg.prt_sigma), 1.0 / problem.dim, 1.0))
                scale = _sample_positive_cauchy(
                    rng,
                    memory_scale[memory_index],
                    cfg.scale_cauchy,
                    lower=0.05,
                    upper=cfg.max_path_scale,
                )
                t = base_t * scale
                mask = sample_mask(rng, problem.dim, p, cfg.force_nonzero_mask)
                raw = origin + t * (basis @ (mask * displacement_basis))
                trial = apply_repair(cfg.repair, raw, origin, lower, upper)
                value = objective(trial)
                proposal_count += 1
                if value <= origin_fit:
                    improvement = max(origin_fit - value, np.finfo(float).tiny)
                    population[migrant_idx] = trial
                    fitness[migrant_idx] = value
                    successful_parameters.append((p, scale, improvement))
                    successful_migrants += 1
                    bin_record["successes"] += 1.0
                    bin_record["selected_p_sum"] += p
                    bin_record["selected_scale_sum"] += scale
                    if value < objective.best_error + np.finfo(float).eps:
                        migration_global_improvement = True
                    accepted = True
                    break

            if not accepted:
                failed_migrant_count += 1

        if successful_parameters:
            p_values = np.array([v[0] for v in successful_parameters], dtype=float)
            scale_values = np.array([v[1] for v in successful_parameters], dtype=float)
            gains = np.array([v[2] for v in successful_parameters], dtype=float)
            weights = gains / np.sum(gains)
            memory_p[memory_cursor] = float(np.sum(weights * p_values))
            memory_scale[memory_cursor] = float(
                np.sum(weights * scale_values**2)
                / max(np.sum(weights * scale_values), np.finfo(float).tiny)
            )
            memory_cursor = (memory_cursor + 1) % cfg.memory_size

        current_best = float(np.min(fitness))
        if migration_global_improvement:
            failed_migrant_count = 0

        # iSOMA-style stagnation restart, adapted to current population size.
        threshold = max(1, int(round(cfg.stagnation_factor * len(population))))
        if failed_migrant_count > threshold and objective.eval_count < max_evals:
            if current_best >= best_cost_old:
                restart_count += 1
                count = max(1, int(round(cfg.restart_fraction * len(population))))
                count = min(count, max(1, len(population) - 1))
                best_idx = int(np.argmin(fitness))
                candidates = np.delete(np.arange(len(population)), best_idx)
                replace_indices = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
                for idx in replace_indices:
                    if objective.eval_count >= max_evals:
                        break
                    candidate = rng.uniform(lower, upper)
                    population[idx] = candidate
                    fitness[idx] = objective(candidate)
            best_cost_old = float(np.min(fitness))
            failed_migrant_count = 0

        # Linear population-size reduction, retaining the best individuals.
        target_size = _linear_population_target(
            initial_size,
            minimum_size,
            objective.eval_count,
            max_evals,
        )
        if len(population) > target_size:
            keep = np.argsort(fitness)[:target_size]
            population = population[keep]
            fitness = fitness[keep]
            reduction_events += 1

    if objective.best_x is None:
        raise RuntimeError("iL-SHOMA-RA performed no valid evaluation.")

    mechanism_rows: list[dict[str, Any]] = []
    for index, record in enumerate(bins):
        if record["migrations"] <= 0:
            continue
        successes = record["successes"]
        mechanism_rows.append(
            {
                "mechanism": "ilshoma_ra_adaptation",
                "progress_bin": index,
                "progress_midpoint": (index + 0.5) / cfg.diagnostic_bins,
                "migration_count": int(record["migrations"]),
                "mean_population_size": record["population_sum"] / record["migrations"],
                "mean_population_fraction": (
                    record["population_sum"] / record["migrations"] / initial_size
                ),
                "migrant_attempts": int(record["attempts"]),
                "successful_migrants": int(successes),
                "migrant_success_rate": successes / max(1.0, record["attempts"]),
                "mean_selected_prt": (
                    record["selected_p_sum"] / successes if successes > 0 else np.nan
                ),
                "mean_selected_path_scale": (
                    record["selected_scale_sum"] / successes if successes > 0 else np.nan
                ),
                "mean_memory_prt": record["memory_p_sum"] / record["migrations"],
                "mean_memory_path_scale": record["memory_scale_sum"] / record["migrations"],
            }
        )

    return OptimizerRun(
        best_x=objective.best_x,
        best_error=float(objective.best_error),
        evals=objective.eval_count,
        history=np.asarray(objective.history_true, dtype=float),
        diagnostics={
            "backend": "native experimental iL-SHOMA-RA",
            "attempted_calls": objective.attempted_calls,
            "migrations": migration,
            "proposals": proposal_count,
            "migrant_attempts": migrant_attempts,
            "successful_migrants": successful_migrants,
            "migrant_success_rate": successful_migrants / max(1, migrant_attempts),
            "initial_population_size": initial_size,
            "minimum_population_size": minimum_size,
            "final_population_size": len(population),
            "population_reduction_events": reduction_events,
            "restart_count": restart_count,
            "memory_p": memory_p.tolist(),
            "memory_path_scale": memory_scale.tolist(),
            "repair": cfg.repair,
        },
        mechanism_rows=mechanism_rows,
    )


# ---------------------------------------------------------------------------
# SciPy DE and iL-SHADE
# ---------------------------------------------------------------------------
def run_scipy_de(
    problem: BBOBProblemAdapter,
    max_evals: int,
    seed: int,
    initial_population: np.ndarray,
) -> OptimizerRun:
    objective = BudgetedObjective(problem, max_evals)
    population_size = len(initial_population)
    remaining = max(0, max_evals - population_size)
    maxiter = int(math.ceil(remaining / population_size)) if remaining else 0
    result = differential_evolution(
        objective,
        problem.bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=max(1, population_size // problem.dim),
        mutation=0.8,
        recombination=0.9,
        seed=int(seed),
        polish=False,
        init=np.asarray(initial_population, dtype=float),
        updating="immediate",
        workers=1,
        tol=0.0,
        atol=0.0,
    )
    best_x = objective.best_x if objective.best_x is not None else np.asarray(result.x, dtype=float)
    best_error = objective.best_error if objective.best_x is not None else max(float(result.fun), 0.0)
    return OptimizerRun(
        best_x=np.asarray(best_x, dtype=float),
        best_error=float(best_error),
        evals=objective.eval_count,
        history=np.asarray(objective.history_true, dtype=float),
        diagnostics={
            "backend": "scipy.optimize.differential_evolution",
            "attempted_calls": objective.attempted_calls,
            "population_size": population_size,
        },
    )


def _ilshade_fallback(
    problem: BBOBProblemAdapter,
    max_evals: int,
    seed: int,
    population_size: int,
) -> OptimizerRun:
    """Development-only iL-SHADE-style fallback; never silently reported."""
    rng = np.random.default_rng(seed)
    lower, upper = problem.lower_bounds, problem.upper_bounds
    objective = BudgetedObjective(problem, max_evals)
    pop = rng.uniform(lower, upper, size=(population_size, problem.dim))
    fit = np.array([objective(x) for x in pop], dtype=float)
    initial_size = population_size
    current_size = population_size
    memory_size = 6
    m_cr = np.full(memory_size, 0.8)
    m_f = np.full(memory_size, 0.5)
    cursor = 0

    while objective.eval_count < max_evals and current_size >= 4:
        indices = rng.integers(memory_size, size=current_size)
        cr = np.clip(rng.normal(m_cr[indices], 0.1), 0.0, 1.0)
        f = np.array([
            _sample_positive_cauchy(rng, m_f[idx], 0.1, 0.0, 1.0)
            for idx in indices
        ])
        p = 0.1 + 0.1 * (objective.eval_count / max_evals)
        order = np.argsort(fit)
        trials = pop.copy()
        trial_fit = np.full(current_size, np.inf)
        for i in range(current_size):
            if objective.eval_count >= max_evals:
                break
            p_count = min(current_size, max(2, int(round(p * current_size))))
            pbest = int(rng.choice(order[:p_count]))
            choices = np.delete(np.arange(current_size), i)
            r1, r2 = rng.choice(choices, size=2, replace=False)
            mutant = pop[i] + f[i] * (pop[pbest] - pop[i]) + f[i] * (pop[r1] - pop[r2])
            mutant = np.clip(mutant, lower, upper)
            mask = rng.random(problem.dim) <= cr[i]
            mask[int(rng.integers(problem.dim))] = True
            trial = np.where(mask, mutant, pop[i])
            trials[i] = trial
            trial_fit[i] = objective(trial)
        improved = np.flatnonzero(trial_fit < fit)
        if improved.size:
            gains = fit[improved] - trial_fit[improved]
            weights = gains / np.sum(gains)
            cr_s, f_s = cr[improved], f[improved]
            m_cr[cursor] = float(np.sum(weights * cr_s))
            m_f[cursor] = float(
                np.sum(weights * f_s**2) / max(np.sum(weights * f_s), np.finfo(float).tiny)
            )
            cursor = (cursor + 1) % memory_size
            pop[improved], fit[improved] = trials[improved], trial_fit[improved]
        new_size = max(4, round(initial_size + (4 - initial_size) * objective.eval_count / max_evals))
        if current_size > new_size:
            keep = np.argsort(fit)[:new_size]
            pop, fit = pop[keep], fit[keep]
            current_size = new_size

    if objective.best_x is None:
        raise RuntimeError("Fallback did not perform a valid evaluation.")
    return OptimizerRun(
        best_x=objective.best_x,
        best_error=float(objective.best_error),
        evals=objective.eval_count,
        history=np.asarray(objective.history_true, dtype=float),
        diagnostics={
            "backend": "local iL-SHADE-style fallback",
            "attempted_calls": objective.attempted_calls,
            "population_size": population_size,
        },
    )


def run_ilshade(
    problem: BBOBProblemAdapter,
    max_evals: int,
    seed: int,
    matched_population_size: Optional[int],
    allow_fallback: bool,
) -> OptimizerRun:
    try:
        from pyade import ilshade  # type: ignore
    except Exception as exc:
        if not allow_fallback:
            raise ModuleNotFoundError(
                "PyADE iL-SHADE could not be imported. Install pyade or pass "
                "--allow-ilshade-fallback for a development-only substitute."
            ) from exc
        fallback_size = matched_population_size or max(18 * problem.dim, 4)
        return _ilshade_fallback(problem, max_evals, seed, fallback_size)

    objective = BudgetedObjective(problem, max_evals)
    params = ilshade.get_default_params(problem.dim)
    default_population = int(params.get("population_size", -1))
    params["max_evals"] = int(max_evals)
    params["bounds"] = np.asarray(problem.bounds, dtype=float)
    params["func"] = objective
    params["seed"] = int(seed)
    if matched_population_size is not None:
        params["population_size"] = int(matched_population_size)
    params["callback"] = None
    params["opts"] = None
    returned_x, returned_f = ilshade.apply(**params)
    best_x = objective.best_x if objective.best_x is not None else np.asarray(returned_x, dtype=float)
    best_error = objective.best_error if objective.best_x is not None else max(float(returned_f), 0.0)
    return OptimizerRun(
        best_x=np.asarray(best_x, dtype=float),
        best_error=float(best_error),
        evals=objective.eval_count,
        history=np.asarray(objective.history_true, dtype=float),
        diagnostics={
            "backend": "PyADE ilshade",
            "attempted_calls": objective.attempted_calls,
            "default_population_size": default_population,
            "effective_population_size": int(params.get("population_size", default_population)),
            "population_mode": "matched" if matched_population_size is not None else "package-default",
        },
    )

# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------
def validate_experiment_config(args: argparse.Namespace) -> None:
    unknown = [name for name in args.algorithms if name not in ALGORITHM_SPECS]
    if unknown:
        raise ValueError(f"Unknown algorithms {unknown}. Available: {list(ALGORITHM_SPECS)}")
    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    if any(d < 2 for d in args.dimensions):
        raise ValueError("All dimensions must be at least 2")
    if any(v < 1 for v in args.budgets_per_dimension):
        raise ValueError("All FE/D budgets must be positive")
    invalid_functions = sorted(set(args.functions).difference(ALL_FUNCTION_IDS))
    if invalid_functions:
        raise ValueError(f"Unsupported BBOB functions: {invalid_functions}")
    if args.matched_population_size is not None and args.matched_population_size < 4:
        raise ValueError("--matched-population-size must be at least 4")
    if args.matched_population_multiplier < 1:
        raise ValueError("--matched-population-multiplier must be at least 1")
    if args.canonical_population_multiplier < 1:
        raise ValueError("--canonical-population-multiplier must be at least 1")
    if args.history_points < 20:
        raise ValueError("--history-points should be at least 20")
    for dimension in args.dimensions:
        matched_size = (
            args.matched_population_size
            if args.matched_population_size is not None
            else args.matched_population_multiplier * dimension
        )
        canonical_size = (
            args.canonical_population_size
            if args.canonical_population_size is not None
            else args.canonical_population_multiplier * dimension
        )
        for budget_pd in args.budgets_per_dimension:
            max_evals = int(round(budget_pd * dimension))
            if max_evals < matched_size:
                raise ValueError(
                    f"Budget {budget_pd} FE/D gives {max_evals} evaluations in D={dimension}, "
                    f"below matched population size {matched_size}."
                )
            if "Canonical-SOMA" in args.algorithms and max_evals < canonical_size:
                raise ValueError(
                    f"Budget {max_evals} is below canonical population size {canonical_size}."
                )


def make_suite_options(
    dimensions: Sequence[int],
    instances: Sequence[int],
    function_ids: Sequence[int],
) -> str:
    return (
        f"function_indices:{','.join(map(str, function_ids))} "
        f"dimensions:{','.join(map(str, dimensions))} "
        f"instance_indices:{','.join(map(str, instances))}"
    )


def run_one_algorithm(
    algorithm: str,
    problem: BBOBProblemAdapter,
    max_evals: int,
    initialization_seed: int,
    algorithm_seed: int,
    matched_initial_population: np.ndarray,
    canonical_initial_population: np.ndarray,
    canonical_cfg: CanonicalSomaConfig,
    operator_cfg: OperatorSomaConfig,
    isoma_class: Optional[type],
    isoma_cfg: ISomaConfig,
    ilshoma_cfg: ILShomaRAConfig,
    allow_ilshade_fallback: bool,
) -> tuple[str, OptimizerRun, int]:
    kind, variant = ALGORITHM_SPECS[algorithm]
    if kind == "canonical_soma":
        result = run_canonical_soma(
            problem,
            max_evals,
            algorithm_seed,
            canonical_initial_population,
            canonical_cfg,
        )
        return algorithm, result, len(canonical_initial_population)
    if kind == "operator_soma":
        result = run_operator_soma(
            problem,
            max_evals,
            algorithm_seed,
            matched_initial_population,
            str(variant),
            operator_cfg,
        )
        return algorithm, result, len(matched_initial_population)
    if kind == "isoma":
        if isoma_class is None:
            raise RuntimeError("iSOMA selected but no class was loaded.")
        # Use the common initialization seed so the supplied np.random.rand(D,N)
        # initialization matches the other matched SOMA algorithms exactly.
        result = run_isoma(
            problem,
            max_evals,
            initialization_seed,
            len(matched_initial_population),
            isoma_class,
            isoma_cfg,
        )
        return algorithm, result, len(matched_initial_population)
    if kind == "ilshoma_ra":
        result = run_ilshoma_ra(
            problem,
            max_evals,
            algorithm_seed,
            matched_initial_population,
            ilshoma_cfg,
        )
        return algorithm, result, len(matched_initial_population)
    if kind == "scipy_de":
        result = run_scipy_de(
            problem,
            max_evals,
            algorithm_seed,
            matched_initial_population,
        )
        return algorithm, result, len(matched_initial_population)
    if kind == "ilshade":
        matched_size = len(matched_initial_population) if variant == "matched" else None
        result = run_ilshade(
            problem,
            max_evals,
            algorithm_seed,
            matched_size,
            allow_fallback=allow_ilshade_fallback,
        )
        fallback = result.diagnostics.get("backend") != "PyADE ilshade"
        if fallback:
            display = "iL-SHADE-matched-fallback" if variant == "matched" else "iL-SHADE-fallback"
        else:
            display = algorithm
        effective_size = int(result.diagnostics.get("effective_population_size", matched_size or -1))
        if effective_size < 0:
            effective_size = int(result.diagnostics.get("population_size", -1))
        return display, result, effective_size
    raise RuntimeError(f"Unhandled algorithm kind: {kind}")


def run_benchmark(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import cocoex
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "COCO's Python interface is required. Install coco-experiment."
        ) from exc

    suite_options = make_suite_options(args.dimensions, args.instances, args.functions)
    suite = cocoex.Suite("bbob", "", suite_options)

    isoma_class: Optional[type] = None
    if "iSOMA" in args.algorithms:
        isoma_class = load_isoma_class(args.isoma_file)

    raw_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []

    total = (
        len(args.dimensions)
        * len(args.budgets_per_dimension)
        * len(args.functions)
        * len(args.instances)
        * len(args.algorithms)
        * args.seeds
    )
    if args.dry_run:
        print(f"Planned optimizer runs: {total}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    done = 0
    experiment_start = time.perf_counter()
    canonical_cfg = CanonicalSomaConfig(
        prt=args.canonical_prt,
        step=args.canonical_step,
        path_length=args.canonical_path_length,
        repair=args.canonical_repair,
        force_nonzero_mask=args.force_nonzero_mask,
    )
    operator_cfg = OperatorSomaConfig(
        n_path_points=args.path_points,
        static_prt=args.operator_prt,
        path_min=args.path_min,
        path_max=args.path_max,
        repair=args.operator_repair,
        force_nonzero_mask=args.force_nonzero_mask,
        geometry_bins=args.diagnostic_bins,
    )
    isoma_cfg = ISomaConfig(
        n_jump=args.isoma_n_jump,
        step=args.isoma_step,
        max_migration=args.isoma_max_migration,
        m=args.isoma_m,
        n=args.isoma_n,
        k=args.isoma_k,
    )
    ilshoma_cfg = ILShomaRAConfig(
        n_jump=args.isoma_n_jump,
        step=args.isoma_step,
        max_migration=args.isoma_max_migration,
        m=args.isoma_m,
        n=args.isoma_n,
        k=args.isoma_k,
        memory_size=args.ilshoma_memory_size,
        prt_sigma=args.ilshoma_prt_sigma,
        scale_cauchy=args.ilshoma_scale_cauchy,
        memory_blend=args.ilshoma_memory_blend,
        min_population_multiplier=args.ilshoma_min_population_multiplier,
        minimum_population=args.ilshoma_minimum_population,
        repair=args.ilshoma_repair,
        force_nonzero_mask=args.force_nonzero_mask,
        stagnation_factor=args.ilshoma_stagnation_factor,
        restart_fraction=args.ilshoma_restart_fraction,
        max_path_scale=args.ilshoma_max_path_scale,
        diagnostic_bins=args.diagnostic_bins,
    )

    try:
        for dimension in args.dimensions:
            matched_population_size = (
                args.matched_population_size
                if args.matched_population_size is not None
                else args.matched_population_multiplier * dimension
            )
            canonical_population_size = (
                args.canonical_population_size
                if args.canonical_population_size is not None
                else args.canonical_population_multiplier * dimension
            )
            for budget_pd in args.budgets_per_dimension:
                max_evals = int(round(budget_pd * dimension))
                for function_id in args.functions:
                    group_id = FUNCTION_TO_GROUP[function_id]
                    group = BBOB_GROUPS[group_id]
                    for instance in args.instances:
                        for repeat in range(args.seeds):
                            initialization_seed = stable_seed(
                                args.base_seed,
                                dimension,
                                budget_pd,
                                function_id,
                                instance,
                                repeat,
                            )
                            # Bounds are invariant for the BBOB problem within this block.
                            temp_problem = suite.get_problem_by_function_dimension_instance(
                                function_id, dimension, instance
                            )
                            try:
                                temp_adapter = BBOBProblemAdapter.from_coco(temp_problem)
                                matched_initial_population = make_legacy_initial_population(
                                    initialization_seed,
                                    matched_population_size,
                                    temp_adapter.lower_bounds,
                                    temp_adapter.upper_bounds,
                                )
                                canonical_initial_population = make_legacy_initial_population(
                                    stable_seed(initialization_seed, 99173),
                                    canonical_population_size,
                                    temp_adapter.lower_bounds,
                                    temp_adapter.upper_bounds,
                                )
                            finally:
                                temp_problem.free()

                            for algorithm_index, algorithm in enumerate(args.algorithms):
                                algorithm_seed = stable_seed(initialization_seed, 1000 + algorithm_index)
                                coco_problem = suite.get_problem_by_function_dimension_instance(
                                    function_id, dimension, instance
                                )
                                try:
                                    problem = BBOBProblemAdapter.from_coco(coco_problem)
                                    start = time.perf_counter()
                                    display_algorithm, result, effective_population_size = run_one_algorithm(
                                        algorithm=algorithm,
                                        problem=problem,
                                        max_evals=max_evals,
                                        initialization_seed=initialization_seed,
                                        algorithm_seed=algorithm_seed,
                                        matched_initial_population=matched_initial_population,
                                        canonical_initial_population=canonical_initial_population,
                                        canonical_cfg=canonical_cfg,
                                        operator_cfg=operator_cfg,
                                        isoma_class=isoma_class,
                                        isoma_cfg=isoma_cfg,
                                        ilshoma_cfg=ilshoma_cfg,
                                        allow_ilshade_fallback=args.allow_ilshade_fallback,
                                    )
                                    elapsed = time.perf_counter() - start
                                    history = np.asarray(result.history, dtype=float)
                                    if result.evals != len(history):
                                        raise RuntimeError(
                                            f"History length {len(history)} != true evaluations {result.evals}."
                                        )
                                    if result.evals > max_evals:
                                        raise RuntimeError(
                                            f"True FE budget exceeded: {result.evals}>{max_evals}."
                                        )

                                    common = {
                                        "group_id": group_id,
                                        "group_number": int(group["number"]),
                                        "group_name": str(group["name"]),
                                        "function_id": function_id,
                                        "function_name": str(group["functions"][function_id]),
                                        "function_label": problem.label,
                                        "instance": instance,
                                        "dimension": dimension,
                                        "budget": max_evals,
                                        "budget_per_dimension": float(budget_pd),
                                        "algorithm": display_algorithm,
                                        "requested_algorithm": algorithm,
                                        "repeat": repeat,
                                        "run_seed": algorithm_seed,
                                        "initialization_seed": initialization_seed,
                                        "population_size": effective_population_size,
                                        "matched_population_size": matched_population_size,
                                        "objective_reference": problem.objective_reference,
                                    }
                                    raw_rows.append(
                                        {
                                            **common,
                                            "evals_used": result.evals,
                                            "attempted_calls": int(
                                                result.diagnostics.get("attempted_calls", result.evals)
                                            ),
                                            "best_error": result.best_error,
                                            "log10_error": error_to_log10(result.best_error),
                                            "absolute_auc_normalized": normalized_absolute_auc(history, max_evals),
                                            "log_auc": normalized_log_auc(history, max_evals),
                                            "elapsed_seconds": elapsed,
                                            "backend": result.diagnostics.get("backend", "native"),
                                            "best_x_json": json.dumps(result.best_x.tolist()),
                                            "diagnostics_json": json.dumps(
                                                result.diagnostics,
                                                default=float,
                                                sort_keys=True,
                                            ),
                                        }
                                    )

                                    grid, sampled = sample_history(
                                        history, max_evals, points=args.history_points
                                    )
                                    for evals, error in zip(grid, sampled):
                                        curve_rows.append(
                                            {
                                                **common,
                                                "evals": int(evals),
                                                "evals_per_dimension": float(evals / dimension),
                                                "best_error": float(error),
                                                "log10_error": error_to_log10(float(error)),
                                            }
                                        )

                                    for target in args.targets:
                                        hit = first_target_hit(history, target)
                                        target_rows.append(
                                            {
                                                **common,
                                                "target_error": target,
                                                "success": hit is not None,
                                                "evals_to_target": int(hit if hit is not None else max_evals),
                                                "evals_to_target_per_dimension": float(
                                                    (hit if hit is not None else max_evals) / dimension
                                                ),
                                            }
                                        )

                                    for mechanism in result.mechanism_rows:
                                        mechanism_rows.append({**common, **mechanism})
                                finally:
                                    coco_problem.free()

                                done += 1
                                if done == 1 or done % args.progress_every == 0 or done == total:
                                    latest = raw_rows[-1]
                                    print(
                                        f"[{done:6d}/{total}] d={dimension:<3d} "
                                        f"budget={budget_pd:g}D ({max_evals:<7d}) "
                                        f"f{function_id:02d}/i{instance:02d} "
                                        f"{latest['algorithm']:<24s} repeat={repeat:<2d} "
                                        f"error={latest['best_error']:.3e} "
                                        f"time={latest['elapsed_seconds']:.2f}s",
                                        flush=True,
                                    )
    finally:
        suite.free()

    raw = pd.DataFrame(raw_rows)
    curves = pd.DataFrame(curve_rows)
    targets = pd.DataFrame(target_rows)
    mechanisms = pd.DataFrame(mechanism_rows)
    raw.to_csv(args.outdir / "benchmark_raw.csv", index=False)
    curves.to_csv(args.outdir / "convergence_curves.csv", index=False)
    targets.to_csv(args.outdir / "target_hits_raw.csv", index=False)
    mechanisms.to_csv(args.outdir / "mechanism_diagnostics.csv", index=False)
    print(f"Benchmark elapsed: {time.perf_counter() - experiment_start:.1f}s", flush=True)
    return raw, curves, targets, mechanisms

# ---------------------------------------------------------------------------
# Publication-oriented summary tables and statistical tests
# ---------------------------------------------------------------------------
def aggregate_final_performance(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dimension",
        "budget",
        "budget_per_dimension",
        "group_id",
        "group_number",
        "group_name",
        "function_id",
        "function_name",
        "algorithm",
    ]
    return (
        raw.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            n_runs=("best_error", "size"),
            median_best_error=("best_error", "median"),
            mean_best_error=("best_error", "mean"),
            std_best_error=("best_error", "std"),
            q25_best_error=("best_error", lambda s: float(np.quantile(s, 0.25))),
            q75_best_error=("best_error", lambda s: float(np.quantile(s, 0.75))),
            min_best_error=("best_error", "min"),
            max_best_error=("best_error", "max"),
            mean_absolute_auc_normalized=("absolute_auc_normalized", "mean"),
            median_absolute_auc_normalized=("absolute_auc_normalized", "median"),
            mean_log10_error=("log10_error", "mean"),
            median_log10_error=("log10_error", "median"),
            mean_log_auc=("log_auc", "mean"),
            median_log_auc=("log_auc", "median"),
            mean_seconds=("elapsed_seconds", "mean"),
            median_seconds=("elapsed_seconds", "median"),
            mean_evals_used=("evals_used", "mean"),
            median_evals_used=("evals_used", "median"),
        )
    )


def make_block_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeats within each function-instance block.

    Ranks are based on the median absolute final error. This is invariant to any
    monotone display transform and avoids using pooled cross-function errors as
    a direct performance score.
    """
    columns = [
        "dimension",
        "budget",
        "budget_per_dimension",
        "group_id",
        "group_number",
        "group_name",
        "function_id",
        "function_name",
        "instance",
        "algorithm",
    ]
    block = (
        raw.groupby(columns, as_index=False, dropna=False)
        .agg(
            block_median_best_error=("best_error", "median"),
            block_mean_best_error=("best_error", "mean"),
            block_q25_best_error=("best_error", lambda s: float(np.quantile(s, 0.25))),
            block_q75_best_error=("best_error", lambda s: float(np.quantile(s, 0.75))),
            block_median_log10_error=("log10_error", "median"),
            block_median_absolute_auc_normalized=("absolute_auc_normalized", "median"),
            block_median_log_auc=("log_auc", "median"),
            repeats=("repeat", "nunique"),
        )
    )
    rank_group = ["dimension", "budget_per_dimension", "function_id", "instance"]
    block["rank"] = block.groupby(rank_group)["block_median_best_error"].rank(
        method="average", ascending=True
    )
    return block


def make_average_ranks(block: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dimension, budget_pd), config in block.groupby(
        ["dimension", "budget_per_dimension"]
    ):
        budget = int(config["budget"].iloc[0])
        scopes: list[tuple[str, Optional[str], pd.DataFrame]] = [("overall", None, config)]
        scopes.extend(("group", gid, subset) for gid, subset in config.groupby("group_id"))
        for scope, group_id, subset in scopes:
            for algorithm, alg_data in subset.groupby("algorithm"):
                rows.append(
                    {
                        "dimension": dimension,
                        "budget": budget,
                        "budget_per_dimension": budget_pd,
                        "scope": scope,
                        "group_id": group_id,
                        "algorithm": algorithm,
                        "n_blocks": alg_data[["function_id", "instance"]].drop_duplicates().shape[0],
                        "average_rank": float(alg_data["rank"].mean()),
                        "median_rank": float(alg_data["rank"].median()),
                        "median_block_best_error": float(
                            alg_data["block_median_best_error"].median()
                        ),
                        "mean_block_absolute_auc_normalized": float(
                            alg_data["block_median_absolute_auc_normalized"].mean()
                        ),
                        "mean_block_log10_error": float(
                            alg_data["block_median_log10_error"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_target_attainment(
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    function_columns = [
        "dimension",
        "budget",
        "budget_per_dimension",
        "group_id",
        "group_number",
        "group_name",
        "function_id",
        "function_name",
        "algorithm",
        "target_error",
    ]
    function_rows: list[dict[str, Any]] = []
    for keys, subset in targets.groupby(function_columns, dropna=False):
        row = dict(zip(function_columns, keys))
        successes = int(subset["success"].sum())
        runs = int(len(subset))
        successful = subset.loc[subset["success"], "evals_to_target"]
        row.update(
            {
                "n_runs": runs,
                "successes": successes,
                "success_rate": successes / runs,
                "ert_evals": (
                    float(subset["evals_to_target"].sum() / successes)
                    if successes
                    else np.inf
                ),
                "ert_evals_per_dimension": (
                    float(subset["evals_to_target_per_dimension"].sum() / successes)
                    if successes
                    else np.inf
                ),
                "median_successful_evals": float(successful.median()) if successes else np.nan,
                "q25_successful_evals": (
                    float(successful.quantile(0.25)) if successes else np.nan
                ),
                "q75_successful_evals": (
                    float(successful.quantile(0.75)) if successes else np.nan
                ),
            }
        )
        function_rows.append(row)

    overall_rows: list[dict[str, Any]] = []
    for (dimension, budget_pd), config in targets.groupby(
        ["dimension", "budget_per_dimension"]
    ):
        budget = int(config["budget"].iloc[0])
        scopes: list[tuple[str, Optional[str], pd.DataFrame]] = [("overall", None, config)]
        scopes.extend(("group", gid, subset) for gid, subset in config.groupby("group_id"))
        for scope, group_id, subset in scopes:
            for (algorithm, target), alg_target in subset.groupby(
                ["algorithm", "target_error"]
            ):
                successes = int(alg_target["success"].sum())
                runs = int(len(alg_target))
                successful = alg_target.loc[alg_target["success"], "evals_to_target"]
                overall_rows.append(
                    {
                        "dimension": dimension,
                        "budget": budget,
                        "budget_per_dimension": budget_pd,
                        "scope": scope,
                        "group_id": group_id,
                        "algorithm": algorithm,
                        "target_error": target,
                        "n_runs": runs,
                        "successes": successes,
                        "success_rate": successes / runs,
                        "ert_evals": (
                            float(alg_target["evals_to_target"].sum() / successes)
                            if successes
                            else np.inf
                        ),
                        "ert_evals_per_dimension": (
                            float(
                                alg_target["evals_to_target_per_dimension"].sum()
                                / successes
                            )
                            if successes
                            else np.inf
                        ),
                        "median_successful_evals": (
                            float(successful.median()) if successes else np.nan
                        ),
                    }
                )
    return pd.DataFrame(function_rows), pd.DataFrame(overall_rows)


def complete_block_pivot(subset: pd.DataFrame, value_column: str) -> pd.DataFrame:
    pivot = subset.pivot_table(
        index=["function_id", "instance"],
        columns="algorithm",
        values=value_column,
        aggfunc="first",
    )
    return pivot.dropna(axis=0, how="any")


def statistical_tests(block: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    friedman_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for (dimension, budget_pd), config in block.groupby(
        ["dimension", "budget_per_dimension"]
    ):
        budget = int(config["budget"].iloc[0])
        scopes: list[tuple[str, Optional[str], pd.DataFrame]] = [("overall", None, config)]
        scopes.extend(("group", gid, subset) for gid, subset in config.groupby("group_id"))
        for scope, group_id, subset in scopes:
            pivot = complete_block_pivot(subset, "block_median_best_error")
            algorithms = list(pivot.columns)
            n_blocks = len(pivot)
            statistic = np.nan
            pvalue = np.nan
            if len(algorithms) >= 3 and n_blocks >= 2:
                result = stats.friedmanchisquare(
                    *[pivot[column].to_numpy() for column in algorithms]
                )
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            friedman_rows.append(
                {
                    "dimension": dimension,
                    "budget": budget,
                    "budget_per_dimension": budget_pd,
                    "scope": scope,
                    "group_id": group_id,
                    "n_blocks": n_blocks,
                    "n_algorithms": len(algorithms),
                    "algorithms": ",".join(algorithms),
                    "friedman_statistic": statistic,
                    "friedman_pvalue": pvalue,
                    "primary_measure": "block median absolute final error",
                }
            )

            local_rows: list[dict[str, Any]] = []
            local_pvalues: list[float] = []
            for algorithm_a, algorithm_b in itertools.combinations(algorithms, 2):
                a = pivot[algorithm_a].to_numpy(dtype=float)
                b = pivot[algorithm_b].to_numpy(dtype=float)
                differences = a - b
                close = np.isclose(a, b, rtol=1e-12, atol=1e-15)
                if np.all(close):
                    wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
                else:
                    try:
                        result = stats.wilcoxon(
                            a,
                            b,
                            zero_method="wilcox",
                            alternative="two-sided",
                            method="auto",
                        )
                        wilcoxon_statistic = float(result.statistic)
                        wilcoxon_pvalue = float(result.pvalue)
                    except ValueError:
                        wilcoxon_statistic, wilcoxon_pvalue = np.nan, 1.0
                wins_a = int(np.sum((a < b) & ~close))
                ties = int(np.sum(close))
                losses_a = int(np.sum((a > b) & ~close))
                log_ratio = np.log10(np.maximum(a, ERROR_FLOOR)) - np.log10(
                    np.maximum(b, ERROR_FLOOR)
                )
                row = {
                    "dimension": dimension,
                    "budget": budget,
                    "budget_per_dimension": budget_pd,
                    "scope": scope,
                    "group_id": group_id,
                    "n_blocks": n_blocks,
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "wins_a": wins_a,
                    "ties": ties,
                    "losses_a": losses_a,
                    "median_absolute_difference_a_minus_b": float(np.median(differences)),
                    "mean_absolute_difference_a_minus_b": float(np.mean(differences)),
                    "median_log10_ratio_a_over_b": float(np.median(log_ratio)),
                    "mean_log10_ratio_a_over_b": float(np.mean(log_ratio)),
                    # Backward-compatible aliases from the previous output.
                    "median_log10_difference_a_minus_b": float(np.median(log_ratio)),
                    "mean_log10_difference_a_minus_b": float(np.mean(log_ratio)),
                    "wilcoxon_statistic": wilcoxon_statistic,
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                    "wilcoxon_measure": "paired block median absolute final errors",
                }
                local_rows.append(row)
                local_pvalues.append(wilcoxon_pvalue)
            for row, adjusted in zip(local_rows, holm_adjust(local_pvalues)):
                row["holm_adjusted_pvalue"] = adjusted
                row["significant_at_0_05"] = bool(adjusted < 0.05)
                pairwise_rows.append(row)
    return pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows)


def summarize_benchmark(
    raw: pd.DataFrame,
    targets: pd.DataFrame,
    outdir: Path,
) -> dict[str, pd.DataFrame]:
    final_summary = aggregate_final_performance(raw)
    block_scores = make_block_scores(raw)
    average_ranks = make_average_ranks(block_scores)
    target_function, target_overall = aggregate_target_attainment(targets)
    friedman, pairwise = statistical_tests(block_scores)

    overall_algorithm = (
        raw.groupby(
            ["dimension", "budget", "budget_per_dimension", "algorithm"],
            as_index=False,
        )
        .agg(
            n_runs=("best_error", "size"),
            n_functions=("function_id", "nunique"),
            n_instances=("instance", "nunique"),
            median_best_error=("best_error", "median"),
            mean_best_error=("best_error", "mean"),
            median_absolute_auc_normalized=("absolute_auc_normalized", "median"),
            mean_absolute_auc_normalized=("absolute_auc_normalized", "mean"),
            mean_log10_error=("log10_error", "mean"),
            median_log10_error=("log10_error", "median"),
            mean_log_auc=("log_auc", "mean"),
            median_log_auc=("log_auc", "median"),
            mean_seconds=("elapsed_seconds", "mean"),
            total_seconds=("elapsed_seconds", "sum"),
        )
    )
    overall_ranks = average_ranks[average_ranks["scope"] == "overall"][
        [
            "dimension",
            "budget_per_dimension",
            "algorithm",
            "average_rank",
            "median_rank",
            "n_blocks",
        ]
    ]
    overall_algorithm = overall_algorithm.merge(
        overall_ranks,
        on=["dimension", "budget_per_dimension", "algorithm"],
        how="left",
    )

    outputs = {
        "final_performance_summary": final_summary,
        "function_instance_block_scores": block_scores,
        "average_ranks": average_ranks,
        "target_attainment_by_function": target_function,
        "target_attainment_overall": target_overall,
        "friedman_tests": friedman,
        "pairwise_wilcoxon_holm": pairwise,
        "overall_algorithm_summary": overall_algorithm,
    }
    for name, frame in outputs.items():
        frame.to_csv(outdir / f"{name}.csv", index=False)
    return outputs


# ---------------------------------------------------------------------------
# Publication figures
# ---------------------------------------------------------------------------
def algorithm_order(values: Iterable[str]) -> list[str]:
    preferred = [
        "Canonical-SOMA",
        "GC-SOMA",
        "RA-SOMA",
        "RA-SOMA-random",
        "RA-SOMA-fixed",
        "iSOMA",
        "iL-SHOMA-RA",
        "SciPy-DE",
        "iL-SHADE"
    ]
    present = set(values)
    return [name for name in preferred if name in present] + sorted(present.difference(preferred))


def save_figure(fig: plt.Figure, path: Path) -> None:
    kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        kwargs["dpi"] = 300
    fig.savefig(path, **kwargs)


def plot_group_convergence(
    curves: pd.DataFrame,
    outdir: Path,
    dimensions: Sequence[int],
    budgets_per_dimension: Sequence[int],
    formats: Sequence[str],
    xscale: str,
    display_floor: float,
) -> list[Path]:
    """One multi-panel figure per BBOB group and dimension/budget setting.

    Each panel is a single function, so absolute errors are never pooled across
    heterogeneous objective scales. Curves show the median and interquartile
    range over instances and repeats.
    """
    paths: list[Path] = []
    for dimension in dimensions:
        for budget_pd in budgets_per_dimension:
            config = curves[
                (curves["dimension"] == dimension)
                & (curves["budget_per_dimension"] == float(budget_pd))
            ]
            if config.empty:
                continue
            algorithms = algorithm_order(config["algorithm"].unique())
            for group_id, group in BBOB_GROUPS.items():
                function_ids = [fid for fid in group["functions"] if fid in set(config["function_id"])]
                if not function_ids:
                    continue
                ncols = 3
                nrows = int(math.ceil(len(function_ids) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(15.5, 4.2 * nrows),
                    sharex=True,
                    sharey=False,
                    squeeze=False,
                )
                axes_flat = axes.ravel()
                legend_handles: list[Any] = []
                legend_labels: list[str] = []
                for panel_index, function_id in enumerate(function_ids):
                    ax = axes_flat[panel_index]
                    function_data = config[config["function_id"] == function_id]
                    for algorithm in algorithms:
                        alg = function_data[function_data["algorithm"] == algorithm]
                        if alg.empty:
                            continue
                        summary = (
                            alg.groupby("evals_per_dimension", as_index=False)["best_error"]
                            .agg(
                                median="median",
                                q25=lambda values: float(np.quantile(values, 0.25)),
                                q75=lambda values: float(np.quantile(values, 0.75)),
                            )
                            .sort_values("evals_per_dimension")
                        )
                        style = PLOT_STYLES.get(algorithm, {})
                        x = summary["evals_per_dimension"].to_numpy()
                        median = np.maximum(summary["median"].to_numpy(), display_floor)
                        q25 = np.maximum(summary["q25"].to_numpy(), display_floor)
                        q75 = np.maximum(summary["q75"].to_numpy(), display_floor)
                        (line,) = ax.plot(
                            x,
                            median,
                            label=algorithm,
                            color=style.get("color"),
                            linestyle=style.get("linestyle", "-"),
                            marker=style.get("marker", None),
                            markevery=max(1, len(summary) // 10),
                            markersize=4.0,
                            linewidth=1.5,
                            markerfacecolor="white",
                            markeredgewidth=0.9,
                        )
                        ax.fill_between(
                            x,
                            q25,
                            q75,
                            color=style.get("color"),
                            alpha=0.10,
                            linewidth=0,
                        )
                        if algorithm not in legend_labels:
                            legend_handles.append(line)
                            legend_labels.append(algorithm)
                    ax.set_title(
                        f"f{function_id:02d} {group['functions'][function_id]}",
                        fontsize=10,
                    )
                    ax.set_yscale("log")
                    ax.set_xscale(xscale)
                    ax.set_xlabel("Function evaluations per dimension")
                    ax.set_ylabel("Best-so-far absolute error")
                    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
                for unused in range(len(function_ids), len(axes_flat)):
                    axes_flat[unused].axis("off")
                fig.suptitle(
                    f"BBOB Group {group['number']}: {group['name']}\n"
                    f"D={dimension}, budget={budget_pd} FE/D; median with IQR",
                    fontsize=14,
                )
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="lower center",
                    ncol=min(4, max(1, len(legend_labels))),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.01),
                )
                fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
                stem = f"convergence_{group['slug']}_d{dimension}_bpd{budget_pd}"
                for fmt in formats:
                    path = outdir / f"{stem}.{fmt}"
                    save_figure(fig, path)
                    paths.append(path)
                plt.close(fig)
    return paths


def plot_target_ecdf(
    targets: pd.DataFrame,
    outdir: Path,
    budgets_per_dimension: Sequence[int],
    formats: Sequence[str],
) -> list[Path]:
    """COCO-style aggregate target-attainment ECDF over all dimensions."""
    paths: list[Path] = []
    for budget_pd in budgets_per_dimension:
        subset = targets[targets["budget_per_dimension"] == float(budget_pd)]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        grid = np.unique(
            np.concatenate(
                ([1.0, float(budget_pd)], np.geomspace(1.0, float(budget_pd), 240))
            )
        )
        for algorithm in algorithm_order(subset["algorithm"].unique()):
            alg = subset[subset["algorithm"] == algorithm]
            hit_times = alg["evals_to_target_per_dimension"].to_numpy(dtype=float)
            successes = alg["success"].to_numpy(dtype=bool)
            values = np.array([
                np.mean(successes & (hit_times <= x)) for x in grid
            ])
            style = PLOT_STYLES.get(algorithm, {})
            ax.plot(
                grid,
                values,
                label=algorithm,
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.8,
            )
        ax.set_xscale("log")
        ax.set_xlim(1.0, float(budget_pd))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Function evaluations per dimension")
        ax.set_ylabel("Fraction of run–target pairs attained")
        ax.set_title(f"BBOB target-attainment ECDF — budget={budget_pd} FE/D")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        for fmt in formats:
            path = outdir / f"target_ecdf_bpd{budget_pd}.{fmt}"
            save_figure(fig, path)
            paths.append(path)
        plt.close(fig)
    return paths


def plot_average_ranks(
    average_ranks: pd.DataFrame,
    outdir: Path,
    budgets_per_dimension: Sequence[int],
    formats: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    overall = average_ranks[average_ranks["scope"] == "overall"]
    for budget_pd in budgets_per_dimension:
        subset = overall[overall["budget_per_dimension"] == float(budget_pd)]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for algorithm in algorithm_order(subset["algorithm"].unique()):
            alg = subset[subset["algorithm"] == algorithm].sort_values("dimension")
            style = PLOT_STYLES.get(algorithm, {})
            ax.plot(
                alg["dimension"],
                alg["average_rank"],
                label=algorithm,
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", "o"),
                linewidth=1.6,
            )
        ax.invert_yaxis()
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Average rank (lower is better)")
        ax.set_title(f"Overall BBOB ranks — budget={budget_pd} FE/D")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        for fmt in formats:
            path = outdir / f"average_ranks_bpd{budget_pd}.{fmt}"
            save_figure(fig, path)
            paths.append(path)
        plt.close(fig)
    return paths


def plot_mechanism_diagnostics(
    mechanisms: pd.DataFrame,
    outdir: Path,
    formats: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    if mechanisms.empty:
        return paths

    geometry = mechanisms[mechanisms["mechanism"] == "geometry_control"]
    if not geometry.empty:
        grouped = geometry.groupby("progress_midpoint", as_index=False).agg(
            target_active_dimension=("target_active_dimension", "mean"),
            empirical_active_dimension=("empirical_active_dimension", "mean"),
            target_contraction=("target_contraction", "mean"),
            empirical_raw_contraction=("empirical_raw_contraction", "mean"),
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(grouped["progress_midpoint"], grouped["target_active_dimension"], label="Target active dimension", linestyle="-")
        ax.plot(grouped["progress_midpoint"], grouped["empirical_active_dimension"], label="Empirical active dimension", linestyle="--")
        ax.set_xlabel("Consumed budget fraction")
        ax.set_ylabel("Active dimensions")
        ax.set_title("GC-SOMA active-dimension control")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        for fmt in formats:
            path = outdir / f"gc_soma_active_dimension.{fmt}"
            save_figure(fig, path)
            paths.append(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(grouped["progress_midpoint"], grouped["target_contraction"], label="Target contraction", linestyle="-")
        ax.plot(grouped["progress_midpoint"], grouped["empirical_raw_contraction"], label="Empirical raw contraction", linestyle="--")
        ax.set_xlabel("Consumed budget fraction")
        ax.set_ylabel("Normalized squared leader distance")
        ax.set_title("GC-SOMA contraction control")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        for fmt in formats:
            path = outdir / f"gc_soma_contraction.{fmt}"
            save_figure(fig, path)
            paths.append(path)
        plt.close(fig)

    adaptation = mechanisms[mechanisms["mechanism"] == "ilshoma_ra_adaptation"]
    if not adaptation.empty:
        grouped = adaptation.groupby("progress_midpoint", as_index=False).agg(
            mean_population_fraction=("mean_population_fraction", "mean"),
            mean_selected_prt=("mean_selected_prt", "mean"),
            mean_selected_path_scale=("mean_selected_path_scale", "mean"),
            mean_memory_prt=("mean_memory_prt", "mean"),
            mean_memory_path_scale=("mean_memory_path_scale", "mean"),
            migrant_success_rate=("migrant_success_rate", "mean"),
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(grouped["progress_midpoint"], grouped["mean_population_fraction"], label="Population fraction")
        ax.plot(grouped["progress_midpoint"], grouped["migrant_success_rate"], label="Migrant success rate", linestyle="--")
        ax.set_xlabel("Consumed budget fraction")
        ax.set_ylabel("Fraction")
        ax.set_ylim(bottom=0.0)
        ax.set_title("iL-SHOMA-RA population reduction and success rate")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        for fmt in formats:
            path = outdir / f"ilshoma_ra_population_success.{fmt}"
            save_figure(fig, path)
            paths.append(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(grouped["progress_midpoint"], grouped["mean_selected_prt"], label="Selected PRT")
        ax.plot(grouped["progress_midpoint"], grouped["mean_memory_prt"], label="Memory PRT", linestyle="--")
        ax.plot(grouped["progress_midpoint"], grouped["mean_selected_path_scale"], label="Selected path scale", linestyle="-.")
        ax.plot(grouped["progress_midpoint"], grouped["mean_memory_path_scale"], label="Memory path scale", linestyle=":")
        ax.set_xlabel("Consumed budget fraction")
        ax.set_ylabel("Adapted parameter value")
        ax.set_title("iL-SHOMA-RA success-history adaptation")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        for fmt in formats:
            path = outdir / f"ilshoma_ra_adaptation.{fmt}"
            save_figure(fig, path)
            paths.append(path)
        plt.close(fig)
    return paths

# ---------------------------------------------------------------------------
# Human-readable publication summary
# ---------------------------------------------------------------------------
def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No data._"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.5g}"
            )
    header = "| " + " | ".join(subset.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_publication_summary(
    args: argparse.Namespace,
    summaries: dict[str, pd.DataFrame],
    figure_paths: Sequence[Path],
) -> None:
    overall = summaries["overall_algorithm_summary"]
    friedman = summaries["friedman_tests"]
    lines = [
        "# Full BBOB SOMA/DE benchmark summary",
        "",
        "## Experimental design",
        "",
        f"- BBOB functions: {args.functions} (default f1--f24), organized into groups 1--5.",
        f"- Dimensions: {args.dimensions}",
        f"- Budgets: {args.budgets_per_dimension} function evaluations per dimension.",
        f"- BBOB instances: {args.instances}",
        f"- Independent repeats per instance: {args.seeds}",
        f"- Requested algorithms: {args.algorithms}",
        f"- Matched initial population: {args.matched_population_size if args.matched_population_size is not None else str(args.matched_population_multiplier) + 'D'}.",
        f"- Canonical SOMA population: {args.canonical_population_size if args.canonical_population_size is not None else str(args.canonical_population_multiplier) + 'D'}.",
        f"- Target errors: {args.targets}",
        "- Absolute final error relative to the BBOB optimum is the primary performance value.",
        "- Repeated runs are aggregated within each function-instance block before ranks and paired tests.",
        "- GC-SOMA, RA-SOMA, iSOMA, iL-SHOMA-RA, and SciPy-DE use a shared initial population within each block when their population sizes match.",
        "- iL-SHADE uses PyADE package defaults; iL-SHADE-matched overrides only the initial population size.",
        "",
        "## Overall algorithm summaries",
        "",
    ]
    for (dimension, budget_pd), config in overall.groupby(
        ["dimension", "budget_per_dimension"]
    ):
        ranked = config.sort_values("average_rank")
        lines.extend(
            [
                f"### Dimension {dimension}, budget {budget_pd:g} FE/D",
                "",
                markdown_table(
                    ranked,
                    [
                        "algorithm",
                        "n_runs",
                        "median_best_error",
                        "median_absolute_auc_normalized",
                        "average_rank",
                        "mean_seconds",
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Friedman tests",
            "",
            markdown_table(
                friedman,
                [
                    "dimension",
                    "budget_per_dimension",
                    "scope",
                    "group_id",
                    "n_blocks",
                    "n_algorithms",
                    "friedman_statistic",
                    "friedman_pvalue",
                ],
            ),
            "",
            "## Generated publication data",
            "",
            "- `benchmark_raw.csv`: one row per optimizer run.",
            "- `convergence_curves.csv`: sampled best-so-far absolute-error histories.",
            "- `target_hits_raw.csv`: run-level target attainment and censored FE counts.",
            "- `mechanism_diagnostics.csv`: GC-SOMA target tracking and iL-SHOMA-RA adaptation/population traces.",
            "- `final_performance_summary.csv`: per-function aggregate error, AUC, and runtime statistics.",
            "- `function_instance_block_scores.csv`: repeated-run medians and absolute-error ranks.",
            "- `average_ranks.csv`: overall and group-specific average ranks.",
            "- `target_attainment_by_function.csv`: success rates and ERT by function.",
            "- `target_attainment_overall.csv`: overall/group success rates and ERT.",
            "- `friedman_tests.csv`: omnibus tests on complete function-instance blocks.",
            "- `pairwise_wilcoxon_holm.csv`: paired tests, Holm correction, effect summaries, and win/tie/loss counts.",
            "- `overall_algorithm_summary.csv`: compact overall performance table.",
            "",
            "## Generated figures",
            "",
        ]
    )
    lines.extend(f"- `{path.name}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "The experimental iL-SHOMA-RA method is a new composite optimizer rather than a published reference implementation. Its results should be presented as exploratory until its components are ablated. Optional RA-SOMA-random and RA-SOMA-fixed runs can be enabled through `--algorithms` to distinguish covariance-informed orientation from generic orthogonal remapping. Any fallback iL-SHADE labels indicate development-only runs and must not be reported as PyADE iL-SHADE.",
        ]
    )
    (args.outdir / "PUBLICATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark canonical and operator-guided SOMA, iSOMA, experimental "
            "iL-SHOMA-RA, DE, and iL-SHADE on official BBOB f1-f24."
        )
    )
    parser.add_argument("--outdir", type=Path, default=Path("bbob_full_results"))
    parser.add_argument(
        "--dimensions",
        type=parse_int_list,
        default=parse_int_list("5,10,20"),
        help="Comma-separated BBOB dimensions.",
    )
    parser.add_argument(
        "--budgets-per-dimension",
        type=parse_int_list,
        default=parse_int_list("500,2500"),
        help="Comma-separated FE/D budgets. Absolute budget is D times this value.",
    )
    parser.add_argument(
        "--functions",
        type=parse_index_spec,
        default=parse_index_spec("1-24"),
        help="BBOB function indices, e.g. 1-24 or 10-24.",
    )
    parser.add_argument(
        "--instances",
        type=parse_index_spec,
        default=parse_index_spec("1-5"),
        help="BBOB instance specification, e.g. 1-5.",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260723)
    parser.add_argument(
        "--algorithms",
        type=parse_name_list,
        default=DEFAULT_ALGORITHMS,
        help="Comma-separated subset of: " + ",".join(ALGORITHM_SPECS),
    )

    parser.add_argument(
        "--matched-population-size",
        type=int,
        default=None,
        help="Fixed common population size for matched algorithms.",
    )
    parser.add_argument(
        "--matched-population-multiplier",
        type=int,
        default=4,
        help="Matched population size = multiplier times D.",
    )
    parser.add_argument(
        "--canonical-population-size",
        type=int,
        default=None,
        help="Fixed canonical SOMA population size.",
    )
    parser.add_argument(
        "--canonical-population-multiplier",
        type=int,
        default=10,
        help="Canonical SOMA population size = multiplier times D.",
    )

    parser.add_argument("--canonical-prt", type=float, default=0.10)
    parser.add_argument("--canonical-step", type=float, default=0.11)
    parser.add_argument("--canonical-path-length", type=float, default=3.0)
    parser.add_argument(
        "--canonical-repair",
        choices=["clip", "reflect", "midpoint"],
        default="clip",
    )

    parser.add_argument("--path-points", type=int, default=6)
    parser.add_argument("--operator-prt", type=float, default=0.30)
    parser.add_argument("--path-min", type=float, default=0.40)
    parser.add_argument("--path-max", type=float, default=2.40)
    parser.add_argument(
        "--operator-repair",
        choices=["clip", "reflect", "midpoint"],
        default="reflect",
    )
    parser.add_argument(
        "--force-nonzero-mask",
        action="store_true",
        help="Condition SOMA masks to contain at least one active coordinate.",
    )

    parser.add_argument(
        "--isoma-file",
        type=Path,
        default=Path("iSOMA.py"),
        help="Path to the supplied file defining ISOMAOptimizer.",
    )
    parser.add_argument("--isoma-n-jump", type=int, default=10)
    parser.add_argument("--isoma-step", type=float, default=0.3)
    parser.add_argument("--isoma-max-migration", type=int, default=10000)
    parser.add_argument("--isoma-m", type=int, default=10)
    parser.add_argument("--isoma-n", type=int, default=5)
    parser.add_argument("--isoma-k", type=int, default=15)

    parser.add_argument("--ilshoma-memory-size", type=int, default=6)
    parser.add_argument("--ilshoma-prt-sigma", type=float, default=0.10)
    parser.add_argument("--ilshoma-scale-cauchy", type=float, default=0.10)
    parser.add_argument("--ilshoma-memory-blend", type=float, default=0.70)
    parser.add_argument("--ilshoma-min-population-multiplier", type=float, default=1.0)
    parser.add_argument("--ilshoma-minimum-population", type=int, default=4)
    parser.add_argument(
        "--ilshoma-repair",
        choices=["clip", "reflect", "midpoint"],
        default="clip",
    )
    parser.add_argument("--ilshoma-stagnation-factor", type=float, default=50.0)
    parser.add_argument("--ilshoma-restart-fraction", type=float, default=0.10)
    parser.add_argument("--ilshoma-max-path-scale", type=float, default=2.0)

    parser.add_argument(
        "--allow-ilshade-fallback",
        action="store_true",
        help="Allow an explicitly labelled development-only fallback if PyADE is unavailable.",
    )
    parser.add_argument(
        "--targets",
        type=parse_float_list,
        default=parse_float_list("1e2,1e1,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8"),
    )
    parser.add_argument("--history-points", type=int, default=201)
    parser.add_argument("--diagnostic-bins", type=int, default=20)
    parser.add_argument(
        "--plot-formats",
        type=parse_name_list,
        default=parse_name_list("png,pdf"),
    )
    parser.add_argument("--plot-xscale", choices=["linear", "log"], default="log")
    parser.add_argument("--plot-error-floor", type=float, default=PLOT_ERROR_FLOOR)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_experiment_config(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["outdir"] = str(args.outdir)
    config["isoma_file"] = str(args.isoma_file)
    config["bbob_groups"] = BBOB_GROUPS
    config["function_ids"] = args.functions
    config["absolute_budgets"] = {
        str(d): [int(d * value) for value in args.budgets_per_dimension]
        for d in args.dimensions
    }
    (args.outdir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    if args.dry_run:
        planned = (
            len(args.dimensions)
            * len(args.budgets_per_dimension)
            * len(args.functions)
            * len(args.instances)
            * len(args.algorithms)
            * args.seeds
        )
        print(f"Planned optimizer runs: {planned}")
        print(f"Configuration written to {(args.outdir / 'config.json').resolve()}")
        return

    raw, curves, targets, mechanisms = run_benchmark(args)
    if args.dry_run:
        return
    summaries = summarize_benchmark(raw, targets, args.outdir)
    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_group_convergence(
            curves=curves,
            outdir=args.outdir,
            dimensions=args.dimensions,
            budgets_per_dimension=args.budgets_per_dimension,
            formats=args.plot_formats,
            xscale=args.plot_xscale,
            display_floor=args.plot_error_floor,
        )
    )
    figure_paths.extend(
        plot_target_ecdf(
            targets=targets,
            outdir=args.outdir,
            budgets_per_dimension=args.budgets_per_dimension,
            formats=args.plot_formats,
        )
    )
    figure_paths.extend(
        plot_average_ranks(
            average_ranks=summaries["average_ranks"],
            outdir=args.outdir,
            budgets_per_dimension=args.budgets_per_dimension,
            formats=args.plot_formats,
        )
    )
    figure_paths.extend(
        plot_mechanism_diagnostics(
            mechanisms=mechanisms,
            outdir=args.outdir,
            formats=args.plot_formats,
        )
    )
    write_publication_summary(args, summaries, figure_paths)
    print(f"Results written to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
