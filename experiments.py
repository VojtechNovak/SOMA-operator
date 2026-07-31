#!/usr/bin/env python3
"""
Numerical validation suite for:

    Linear Proposal Operators and Stochastic Search Geometry
    in SOMA and Differential Evolution

The script validates:
  1. SOMA independent Bernoulli-mask moments.
  2. SOMA masks conditioned on at least one active coordinate.
  3. SOMA path-wide coordinate coverage.
  4. Finite-population DE/rand/1 donor moments.
  5. Exact forced-coordinate DE/bin crossover moments.
  6. Combined donor-marginalized DE trial moments.
  7. Boundary-repair distortion for SOMA and DE.
  8. Pre-selection versus post-selection proposal geometry.

Outputs
-------
<output>/
    metadata.json
    combined_summary.csv
    vector_entries.csv
    matrix_entries.csv
    soma_independent.csv
    soma_conditioned.csv
    soma_coverage.csv
    de_donor.csv
    de_forced_crossover.csv
    de_total.csv
    boundary_repair.csv
    selection_distortion.csv
    figures/*.png
    figures/*.pdf
    RUN_COMPLETE.txt
<output>.zip

Usage
-----
Quick run:
    python run_operator_geometry_experiments.py --mode quick

Full paper-quality run:
    python run_operator_geometry_experiments.py --mode full

Very small installation/smoke test:
    python run_operator_geometry_experiments.py --mode smoke

The script requires only NumPy, pandas, and Matplotlib.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-15


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    mode: str
    seed: int
    mc_samples: int
    coverage_trials: int
    selection_trials: int
    boundary_samples: int
    batch_size: int
    pair_replicates: int
    convergence_samples: tuple[int, ...]


def build_config(mode: str, seed: int) -> RunConfig:
    if mode == "smoke":
        return RunConfig(
            mode=mode,
            seed=seed,
            mc_samples=2_000,
            coverage_trials=2_000,
            selection_trials=1_000,
            boundary_samples=2_000,
            batch_size=500,
            pair_replicates=1,
            convergence_samples=(250, 1_000),
        )

    if mode == "quick":
        return RunConfig(
            mode=mode,
            seed=seed,
            mc_samples=20_000,
            coverage_trials=10_000,
            selection_trials=8_000,
            boundary_samples=20_000,
            batch_size=2_000,
            pair_replicates=2,
            convergence_samples=(500, 2_000, 10_000, 20_000),
        )

    if mode == "full":
        return RunConfig(
            mode=mode,
            seed=seed,
            mc_samples=150_000,
            coverage_trials=60_000,
            selection_trials=50_000,
            boundary_samples=150_000,
            batch_size=5_000,
            pair_replicates=5,
            convergence_samples=(1_000, 5_000, 20_000, 75_000, 150_000),
        )

    raise ValueError(f"Unsupported mode: {mode}")


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_relative_error(empirical: float, theoretical: float) -> float:
    return abs(empirical - theoretical) / max(abs(theoretical), EPS)


def vector_relative_error(empirical: np.ndarray, theoretical: np.ndarray) -> float:
    return float(
        np.linalg.norm(empirical - theoretical)
        / max(np.linalg.norm(theoretical), EPS)
    )


def vector_scaled_error(
    empirical: np.ndarray,
    theoretical: np.ndarray,
    covariance_scale: np.ndarray,
) -> float:
    """
    Mean error normalized by the distribution's root-total variance.

    This remains stable when the theoretical mean is close to zero, where a
    conventional relative error can be arbitrarily large despite a small
    absolute discrepancy.
    """
    denominator = math.sqrt(max(float(np.trace(covariance_scale)), EPS))
    return float(np.linalg.norm(empirical - theoretical) / denominator)


def matrix_relative_error(empirical: np.ndarray, theoretical: np.ndarray) -> float:
    return float(
        np.linalg.norm(empirical - theoretical, ord="fro")
        / max(np.linalg.norm(theoretical, ord="fro"), EPS)
    )


def empirical_mean_cov(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / len(points)
    return mean, covariance


def weighted_mean_cov(
    points: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - mean
    covariance = (centered * weights[:, None]).T @ centered
    return mean, covariance


class MomentAccumulator:
    """Streaming population mean and covariance accumulator."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.count = 0
        self.sum = np.zeros(dimension, dtype=float)
        self.sum_outer = np.zeros((dimension, dimension), dtype=float)

    def update(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return
        points = points.reshape(-1, self.dimension)
        self.count += len(points)
        self.sum += points.sum(axis=0)
        self.sum_outer += points.T @ points

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("No observations were accumulated.")
        mean = self.sum / self.count
        covariance = self.sum_outer / self.count - np.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        return mean, covariance


def add_scalar_row(
    rows: list[dict],
    experiment: str,
    case_id: str,
    metric: str,
    empirical: float,
    theoretical: float | None = None,
    **parameters,
) -> None:
    row = {
        "experiment": experiment,
        "case_id": case_id,
        "metric": metric,
        "empirical": float(empirical),
        "theoretical": np.nan if theoretical is None else float(theoretical),
        "absolute_error": (
            np.nan if theoretical is None else abs(float(empirical) - float(theoretical))
        ),
        "relative_error": (
            np.nan
            if theoretical is None
            else safe_relative_error(float(empirical), float(theoretical))
        ),
    }
    row.update(parameters)
    rows.append(row)


def add_vector_entries(
    rows: list[dict],
    experiment: str,
    case_id: str,
    quantity: str,
    empirical: np.ndarray,
    theoretical: np.ndarray,
    **parameters,
) -> None:
    for index, (emp, theory) in enumerate(zip(empirical, theoretical)):
        row = {
            "experiment": experiment,
            "case_id": case_id,
            "quantity": quantity,
            "index": index,
            "empirical": float(emp),
            "theoretical": float(theory),
            "absolute_error": abs(float(emp) - float(theory)),
        }
        row.update(parameters)
        rows.append(row)


def add_matrix_entries(
    rows: list[dict],
    experiment: str,
    case_id: str,
    quantity: str,
    empirical: np.ndarray,
    theoretical: np.ndarray,
    **parameters,
) -> None:
    dimension = empirical.shape[0]
    for row_index in range(dimension):
        for column_index in range(dimension):
            row = {
                "experiment": experiment,
                "case_id": case_id,
                "quantity": quantity,
                "row": row_index,
                "column": column_index,
                "empirical": float(empirical[row_index, column_index]),
                "theoretical": float(theoretical[row_index, column_index]),
                "absolute_error": abs(
                    float(empirical[row_index, column_index])
                    - float(theoretical[row_index, column_index])
                ),
            }
            row.update(parameters)
            rows.append(row)


def max_abs_off_diagonal(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(matrix[mask])))


def principal_angle_degrees(
    covariance_a: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    _, vectors_a = np.linalg.eigh(covariance_a)
    _, vectors_b = np.linalg.eigh(covariance_b)
    vector_a = vectors_a[:, -1]
    vector_b = vectors_b[:, -1]
    cosine = float(np.clip(abs(np.dot(vector_a, vector_b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def covariance_condition_number(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return float("inf")
    return float(eigenvalues.max() / eigenvalues.min())


def random_migrant_leader_pair(
    dimension: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(0.0, 1.0, size=dimension)
    displacement = rng.normal(0.0, 1.0, size=dimension)

    # Avoid nearly zero displacement coordinates, which make relative-error
    # diagnostics unnecessarily unstable.
    tiny = np.abs(displacement) < 0.15
    displacement[tiny] += np.where(displacement[tiny] >= 0.0, 0.25, -0.25)
    leader = x + displacement
    return x, leader


def make_correlated_population(
    population_size: int,
    dimension: int,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> np.ndarray:
    raw = rng.normal(size=(population_size, dimension))

    random_matrix = rng.normal(size=(dimension, dimension))
    orthogonal, _ = np.linalg.qr(random_matrix)
    eigenvalues = np.linspace(2.0, 0.25, dimension)
    transform = orthogonal @ np.diag(np.sqrt(eigenvalues))

    population = raw @ transform.T
    population -= population.mean(axis=0)
    population *= scale
    return population


def iter_batches(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        yield start, end
        start = end


# =============================================================================
# SOMA theory and sampling
# =============================================================================

def sample_bernoulli_masks(
    n_samples: int,
    dimension: int,
    p: float,
    rng: np.random.Generator,
    conditioned_nonzero: bool = False,
) -> np.ndarray:
    if not conditioned_nonzero:
        return rng.random((n_samples, dimension)) < p

    if p <= 0.0:
        raise ValueError("Conditioning on K>0 is undefined for p=0.")

    accepted: list[np.ndarray] = []
    count = 0
    acceptance = 1.0 - (1.0 - p) ** dimension
    batch_guess = max(1_000, int((n_samples - count) / max(acceptance, 1e-6) * 1.1))

    while count < n_samples:
        batch = rng.random((batch_guess, dimension)) < p
        batch = batch[np.any(batch, axis=1)]
        needed = n_samples - count
        if len(batch) > needed:
            batch = batch[:needed]
        accepted.append(batch)
        count += len(batch)

    return np.vstack(accepted)


def soma_independent_theory(
    x: np.ndarray,
    leader: np.ndarray,
    t: float,
    p: float,
) -> dict[str, np.ndarray | float]:
    r = leader - x
    mean = x + p * t * r
    covariance = p * (1.0 - p) * t**2 * np.diag(r**2)
    expected_step_sq = p * t**2 * float(np.dot(r, r))
    expected_leader_sq = (
        1.0 - p * t * (2.0 - t)
    ) * float(np.dot(r, r))
    return {
        "mean": mean,
        "covariance": covariance,
        "expected_step_sq": expected_step_sq,
        "expected_leader_sq": expected_leader_sq,
    }


def soma_conditioned_mask_moments(
    dimension: int,
    p: float,
) -> tuple[float, np.ndarray]:
    """
    Exact mask moments conditional on K>0.

    alpha = P(pi_d=1 | K>0)
    Gamma = Cov(pi | K>0)
    """
    acceptance = 1.0 - (1.0 - p) ** dimension
    alpha = p / acceptance
    off_diagonal = -p**2 * (1.0 - p) ** dimension / acceptance**2

    gamma = np.full((dimension, dimension), off_diagonal, dtype=float)
    np.fill_diagonal(gamma, alpha * (1.0 - alpha))
    return alpha, gamma


def soma_conditioned_theory(
    x: np.ndarray,
    leader: np.ndarray,
    t: float,
    p: float,
) -> dict[str, np.ndarray | float]:
    dimension = len(x)
    r = leader - x
    alpha, gamma = soma_conditioned_mask_moments(dimension, p)
    diagonal_r = np.diag(r)
    mean = x + t * alpha * r
    covariance = t**2 * diagonal_r @ gamma @ diagonal_r
    return {
        "alpha": alpha,
        "gamma": gamma,
        "mean": mean,
        "covariance": covariance,
    }


# =============================================================================
# DE theory and sampling
# =============================================================================

def de_donor_pool(
    population: np.ndarray,
    target_index: int,
) -> np.ndarray:
    return np.delete(population, target_index, axis=0)


def de_donor_theory(
    population: np.ndarray,
    target_index: int,
    differential_weight: float,
) -> dict[str, np.ndarray | float]:
    pool = de_donor_pool(population, target_index)
    donor_pool_size = len(pool)
    mean = pool.mean(axis=0)
    centered = pool - mean

    # Population covariance convention with denominator M.
    pool_covariance = centered.T @ centered / donor_pool_size
    multiplier = 1.0 + 2.0 * differential_weight**2 * (
        donor_pool_size / (donor_pool_size - 1.0)
    )
    mutant_covariance = multiplier * pool_covariance

    return {
        "mean": mean,
        "pool_covariance": pool_covariance,
        "covariance": mutant_covariance,
        "multiplier": multiplier,
        "pool_size": donor_pool_size,
    }


def enumerate_de_mutants(
    population: np.ndarray,
    target_index: int,
    differential_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    pool = de_donor_pool(population, target_index)
    donor_triples = list(itertools.permutations(range(len(pool)), 3))

    mutants = np.empty((len(donor_triples), population.shape[1]), dtype=float)
    for index, (r1, r2, r3) in enumerate(donor_triples):
        mutants[index] = (
            pool[r1]
            + differential_weight * (pool[r2] - pool[r3])
        )

    weights = np.full(len(mutants), 1.0 / len(mutants), dtype=float)
    return mutants, weights


def sample_distinct_pool_indices(
    n_samples: int,
    pool_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = rng.integers(0, pool_size, size=n_samples)
    r2 = rng.integers(0, pool_size, size=n_samples)

    collision = r2 == r1
    while np.any(collision):
        r2[collision] = rng.integers(0, pool_size, size=np.count_nonzero(collision))
        collision = r2 == r1

    r3 = rng.integers(0, pool_size, size=n_samples)
    collision = (r3 == r1) | (r3 == r2)
    while np.any(collision):
        r3[collision] = rng.integers(0, pool_size, size=np.count_nonzero(collision))
        collision = (r3 == r1) | (r3 == r2)

    return r1, r2, r3


def sample_de_mutants(
    population: np.ndarray,
    target_index: int,
    differential_weight: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = de_donor_pool(population, target_index)
    r1, r2, r3 = sample_distinct_pool_indices(n_samples, len(pool), rng)
    return pool[r1] + differential_weight * (pool[r2] - pool[r3])


def forced_crossover_mask_theory(
    dimension: int,
    crossover_rate: float,
) -> tuple[float, np.ndarray]:
    """
    Exact moments of standard DE/bin crossover after marginalizing j_rand.
    """
    q = crossover_rate
    alpha = q + (1.0 - q) / dimension
    off_diagonal = -(1.0 - q) ** 2 / dimension**2

    gamma = np.full((dimension, dimension), off_diagonal, dtype=float)
    np.fill_diagonal(gamma, alpha * (1.0 - alpha))
    return alpha, gamma


def sample_forced_crossover_masks(
    n_samples: int,
    dimension: int,
    crossover_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    masks = rng.random((n_samples, dimension)) < crossover_rate
    forced = rng.integers(0, dimension, size=n_samples)
    masks[np.arange(n_samples), forced] = True
    return masks


def enumerate_forced_crossover_masks(
    dimension: int,
    crossover_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate exact mask probabilities over the random forced coordinate.
    """
    q = crossover_rate
    probabilities: dict[tuple[int, ...], float] = {}

    for forced_coordinate in range(dimension):
        free_coordinates = [
            coordinate
            for coordinate in range(dimension)
            if coordinate != forced_coordinate
        ]

        for bits in itertools.product([0, 1], repeat=dimension - 1):
            mask = np.zeros(dimension, dtype=int)
            mask[forced_coordinate] = 1
            mask[free_coordinates] = bits

            active_free = int(sum(bits))
            probability = (
                1.0 / dimension
                * q**active_free
                * (1.0 - q) ** (dimension - 1 - active_free)
            )
            key = tuple(mask.tolist())
            probabilities[key] = probabilities.get(key, 0.0) + probability

    masks = np.array(list(probabilities.keys()), dtype=float)
    weights = np.array(list(probabilities.values()), dtype=float)
    weights /= weights.sum()
    return masks, weights


def de_forced_conditional_theory(
    target: np.ndarray,
    mutant: np.ndarray,
    crossover_rate: float,
) -> dict[str, np.ndarray | float]:
    delta = mutant - target
    alpha, gamma = forced_crossover_mask_theory(
        len(target),
        crossover_rate,
    )
    diagonal_delta = np.diag(delta)
    mean = target + alpha * delta
    covariance = diagonal_delta @ gamma @ diagonal_delta
    return {
        "alpha": alpha,
        "gamma": gamma,
        "mean": mean,
        "covariance": covariance,
    }


def de_total_theory(
    population: np.ndarray,
    target_index: int,
    differential_weight: float,
    crossover_rate: float,
) -> dict[str, np.ndarray | float]:
    target = population[target_index]
    donor = de_donor_theory(population, target_index, differential_weight)
    mutant_mean = np.asarray(donor["mean"])
    mutant_covariance = np.asarray(donor["covariance"])
    delta = mutant_mean - target

    alpha, gamma = forced_crossover_mask_theory(
        population.shape[1],
        crossover_rate,
    )

    trial_mean = target + alpha * delta
    second_moment_about_target = mutant_covariance + np.outer(delta, delta)
    trial_covariance = (
        alpha**2 * mutant_covariance
        + gamma * second_moment_about_target
    )

    return {
        "target": target,
        "mutant_mean": mutant_mean,
        "mutant_covariance": mutant_covariance,
        "trial_mean": trial_mean,
        "trial_covariance": trial_covariance,
        "alpha": alpha,
        "gamma": gamma,
    }


def sample_de_trials(
    population: np.ndarray,
    target_index: int,
    differential_weight: float,
    crossover_rate: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mutants = sample_de_mutants(
        population,
        target_index,
        differential_weight,
        n_samples,
        rng,
    )
    masks = sample_forced_crossover_masks(
        n_samples,
        population.shape[1],
        crossover_rate,
        rng,
    )
    target = population[target_index]
    trials = target + masks * (mutants - target)
    return mutants, trials


# =============================================================================
# Boundary repair
# =============================================================================

def clip_repair(points: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(points, lower, upper)


def reflect_repair(points: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Repeated reflection into [lower, upper], valid for arbitrary overshoot.
    """
    width = upper - lower
    shifted = np.mod(points - lower, 2.0 * width)
    reflected = np.where(shifted <= width, shifted, 2.0 * width - shifted)
    return lower + reflected


def boundary_distortion_metrics(
    raw: np.ndarray,
    repaired: np.ndarray,
    lower: float,
    upper: float,
) -> dict[str, float]:
    raw_mean, raw_covariance = empirical_mean_cov(raw)
    repaired_mean, repaired_covariance = empirical_mean_cov(repaired)

    outside = np.any((raw < lower) | (raw > upper), axis=1)
    return {
        "repair_fraction": float(outside.mean()),
        "mean_shift_l2": float(np.linalg.norm(repaired_mean - raw_mean)),
        "covariance_shift_fro": float(
            np.linalg.norm(repaired_covariance - raw_covariance, ord="fro")
        ),
        "raw_covariance_trace": float(np.trace(raw_covariance)),
        "repaired_covariance_trace": float(np.trace(repaired_covariance)),
    }


# =============================================================================
# Objective functions for selection experiments
# =============================================================================

def sphere_values(points: np.ndarray) -> np.ndarray:
    return np.sum(points**2, axis=-1)


def make_rotated_quadratic(
    dimension: int,
    rng: np.random.Generator,
) -> Callable[[np.ndarray], np.ndarray]:
    random_matrix = rng.normal(size=(dimension, dimension))
    orthogonal, _ = np.linalg.qr(random_matrix)
    eigenvalues = np.geomspace(1.0, 100.0, dimension)
    hessian = orthogonal @ np.diag(eigenvalues) @ orthogonal.T

    def objective(points: np.ndarray) -> np.ndarray:
        return np.einsum("...i,ij,...j->...", points, hessian, points)

    return objective


# =============================================================================
# Experiment 1: independent SOMA moments
# =============================================================================

def experiment_soma_independent(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
    vectors: list[dict],
    matrices: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []

    dimensions = [3, 10] if config.mode == "smoke" else [3, 10, 30]
    probabilities = [0.2, 0.5] if config.mode == "smoke" else [0.1, 0.3, 0.5, 0.8]
    path_parameters = [0.5, 1.5] if config.mode == "smoke" else [0.5, 1.0, 1.5, 2.5]

    for dimension in dimensions:
        for replicate in range(config.pair_replicates):
            x, leader = random_migrant_leader_pair(dimension, rng)
            r = leader - x

            for p in probabilities:
                for t in path_parameters:
                    case_id = f"D{dimension}_rep{replicate}_p{p}_t{t}"
                    masks = sample_bernoulli_masks(
                        config.mc_samples,
                        dimension,
                        p,
                        rng,
                    ).astype(float)
                    proposals = x + t * masks * r

                    empirical_mean, empirical_covariance = empirical_mean_cov(proposals)
                    theory = soma_independent_theory(x, leader, t, p)
                    theoretical_mean = np.asarray(theory["mean"])
                    theoretical_covariance = np.asarray(theory["covariance"])

                    empirical_step_sq = float(
                        np.mean(np.sum((proposals - x) ** 2, axis=1))
                    )
                    empirical_leader_sq = float(
                        np.mean(np.sum((proposals - leader) ** 2, axis=1))
                    )

                    parameters = {
                        "D": dimension,
                        "replicate": replicate,
                        "p": p,
                        "t": t,
                        "n_samples": config.mc_samples,
                    }

                    metrics = {
                        "mean_relative_l2_error": vector_relative_error(
                            empirical_mean,
                            theoretical_mean,
                        ),
                        "covariance_relative_fro_error": matrix_relative_error(
                            empirical_covariance,
                            theoretical_covariance,
                        ),
                        "empirical_covariance_max_abs_offdiag": max_abs_off_diagonal(
                            empirical_covariance
                        ),
                    }

                    for metric, value in metrics.items():
                        add_scalar_row(
                            summary,
                            "soma_independent",
                            case_id,
                            metric,
                            value,
                            None,
                            **parameters,
                        )

                    add_scalar_row(
                        summary,
                        "soma_independent",
                        case_id,
                        "expected_step_squared",
                        empirical_step_sq,
                        float(theory["expected_step_sq"]),
                        **parameters,
                    )
                    add_scalar_row(
                        summary,
                        "soma_independent",
                        case_id,
                        "expected_leader_distance_squared",
                        empirical_leader_sq,
                        float(theory["expected_leader_sq"]),
                        **parameters,
                    )

                    add_vector_entries(
                        vectors,
                        "soma_independent",
                        case_id,
                        "proposal_mean",
                        empirical_mean,
                        theoretical_mean,
                        **parameters,
                    )
                    add_matrix_entries(
                        matrices,
                        "soma_independent",
                        case_id,
                        "proposal_covariance",
                        empirical_covariance,
                        theoretical_covariance,
                        **parameters,
                    )

                    records.append({
                        "case_id": case_id,
                        **parameters,
                        **metrics,
                        "expected_step_squared_empirical": empirical_step_sq,
                        "expected_step_squared_theoretical": float(
                            theory["expected_step_sq"]
                        ),
                        "expected_leader_squared_empirical": empirical_leader_sq,
                        "expected_leader_squared_theoretical": float(
                            theory["expected_leader_sq"]
                        ),
                    })

    # Representative Monte Carlo convergence case.
    dimension = 10
    x, leader = random_migrant_leader_pair(dimension, rng)
    p, t = 0.3, 1.5
    theory = soma_independent_theory(x, leader, t, p)

    for n_samples in config.convergence_samples:
        masks = sample_bernoulli_masks(
            n_samples,
            dimension,
            p,
            rng,
        ).astype(float)
        proposals = x + t * masks * (leader - x)
        mean, covariance = empirical_mean_cov(proposals)

        records.append({
            "case_id": f"convergence_n{n_samples}",
            "D": dimension,
            "replicate": -1,
            "p": p,
            "t": t,
            "n_samples": n_samples,
            "mean_relative_l2_error": vector_relative_error(
                mean,
                np.asarray(theory["mean"]),
            ),
            "covariance_relative_fro_error": matrix_relative_error(
                covariance,
                np.asarray(theory["covariance"]),
            ),
            "empirical_covariance_max_abs_offdiag": max_abs_off_diagonal(covariance),
            "expected_step_squared_empirical": np.nan,
            "expected_step_squared_theoretical": np.nan,
            "expected_leader_squared_empirical": np.nan,
            "expected_leader_squared_theoretical": np.nan,
            "is_convergence_case": True,
        })

    return pd.DataFrame(records)


# =============================================================================
# Experiment 2: conditioned SOMA masks
# =============================================================================

def experiment_soma_conditioned(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
    vectors: list[dict],
    matrices: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []

    dimensions = [3, 10] if config.mode == "smoke" else [3, 10, 30]
    probabilities = [0.2, 0.5] if config.mode == "smoke" else [0.05, 0.2, 0.5]
    t = 1.0

    for dimension in dimensions:
        x, leader = random_migrant_leader_pair(dimension, rng)
        r = leader - x

        for p in probabilities:
            case_id = f"D{dimension}_p{p}"
            masks = sample_bernoulli_masks(
                config.mc_samples,
                dimension,
                p,
                rng,
                conditioned_nonzero=True,
            ).astype(float)
            proposals = x + t * masks * r

            mask_mean, mask_covariance = empirical_mean_cov(masks)
            proposal_mean, proposal_covariance = empirical_mean_cov(proposals)
            theory = soma_conditioned_theory(x, leader, t, p)

            alpha = float(theory["alpha"])
            gamma = np.asarray(theory["gamma"])
            theoretical_mean = np.asarray(theory["mean"])
            theoretical_covariance = np.asarray(theory["covariance"])

            unconditioned = soma_independent_theory(x, leader, t, p)
            unconditioned_covariance = np.asarray(unconditioned["covariance"])

            parameters = {
                "D": dimension,
                "p": p,
                "t": t,
                "n_samples": config.mc_samples,
            }

            add_scalar_row(
                summary,
                "soma_conditioned",
                case_id,
                "mask_marginal_activation",
                float(mask_mean.mean()),
                alpha,
                **parameters,
            )
            add_scalar_row(
                summary,
                "soma_conditioned",
                case_id,
                "mask_covariance_relative_fro_error",
                matrix_relative_error(mask_covariance, gamma),
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "soma_conditioned",
                case_id,
                "proposal_mean_relative_l2_error",
                vector_relative_error(proposal_mean, theoretical_mean),
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "soma_conditioned",
                case_id,
                "proposal_covariance_relative_fro_error",
                matrix_relative_error(
                    proposal_covariance,
                    theoretical_covariance,
                ),
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "soma_conditioned",
                case_id,
                "theoretical_mask_offdiagonal",
                float(gamma[0, 1]) if dimension > 1 else 0.0,
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "soma_conditioned",
                case_id,
                "conditioned_vs_unconditioned_covariance_fro",
                float(
                    np.linalg.norm(
                        theoretical_covariance - unconditioned_covariance,
                        ord="fro",
                    )
                ),
                None,
                **parameters,
            )

            add_vector_entries(
                vectors,
                "soma_conditioned",
                case_id,
                "proposal_mean",
                proposal_mean,
                theoretical_mean,
                **parameters,
            )
            add_matrix_entries(
                matrices,
                "soma_conditioned",
                case_id,
                "mask_covariance",
                mask_covariance,
                gamma,
                **parameters,
            )
            add_matrix_entries(
                matrices,
                "soma_conditioned",
                case_id,
                "proposal_covariance",
                proposal_covariance,
                theoretical_covariance,
                **parameters,
            )

            records.append({
                "case_id": case_id,
                **parameters,
                "mask_activation_empirical": float(mask_mean.mean()),
                "mask_activation_theoretical": alpha,
                "mask_covariance_relative_fro_error": matrix_relative_error(
                    mask_covariance,
                    gamma,
                ),
                "proposal_mean_relative_l2_error": vector_relative_error(
                    proposal_mean,
                    theoretical_mean,
                ),
                "proposal_covariance_relative_fro_error": matrix_relative_error(
                    proposal_covariance,
                    theoretical_covariance,
                ),
                "mask_offdiagonal_empirical_mean": (
                    float(mask_covariance[~np.eye(dimension, dtype=bool)].mean())
                    if dimension > 1
                    else 0.0
                ),
                "mask_offdiagonal_theoretical": (
                    float(gamma[0, 1]) if dimension > 1 else 0.0
                ),
                "conditioned_vs_unconditioned_covariance_fro": float(
                    np.linalg.norm(
                        theoretical_covariance - unconditioned_covariance,
                        ord="fro",
                    )
                ),
            })

    return pd.DataFrame(records)


# =============================================================================
# Experiment 3: path coverage
# =============================================================================

def experiment_soma_coverage(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []

    dimensions = [10, 30] if config.mode == "smoke" else [10, 30, 100]
    probabilities = [0.1, 0.3] if config.mode == "smoke" else [0.05, 0.1, 0.2, 0.5]
    path_lengths = [1, 5] if config.mode == "smoke" else [1, 2, 5, 10, 20, 50]

    for dimension in dimensions:
        for p in probabilities:
            for path_length in path_lengths:
                for mask_mode in ("independent", "reused"):
                    uncovered_total = 0.0
                    all_covered_total = 0
                    covered_fraction_total = 0.0

                    for start, end in iter_batches(
                        config.coverage_trials,
                        config.batch_size,
                    ):
                        batch = end - start

                        if mask_mode == "independent":
                            masks = (
                                rng.random((batch, path_length, dimension)) < p
                            )
                            covered = np.any(masks, axis=1)
                            one_coordinate_probability = 1.0 - (1.0 - p) ** path_length
                        else:
                            mask = rng.random((batch, dimension)) < p
                            covered = mask
                            one_coordinate_probability = p

                        uncovered = dimension - covered.sum(axis=1)
                        uncovered_total += float(uncovered.sum())
                        all_covered_total += int(np.all(covered, axis=1).sum())
                        covered_fraction_total += float(covered.mean(axis=1).sum())

                    empirical_uncovered = uncovered_total / config.coverage_trials
                    empirical_all_covered = all_covered_total / config.coverage_trials
                    empirical_coordinate_coverage = (
                        covered_fraction_total / config.coverage_trials
                    )

                    theoretical_uncovered = dimension * (
                        1.0 - one_coordinate_probability
                    )
                    theoretical_all_covered = one_coordinate_probability**dimension

                    case_id = (
                        f"D{dimension}_p{p}_J{path_length}_{mask_mode}"
                    )
                    parameters = {
                        "D": dimension,
                        "p": p,
                        "J": path_length,
                        "mask_mode": mask_mode,
                        "n_trials": config.coverage_trials,
                    }

                    add_scalar_row(
                        summary,
                        "soma_coverage",
                        case_id,
                        "coordinate_coverage_probability",
                        empirical_coordinate_coverage,
                        one_coordinate_probability,
                        **parameters,
                    )
                    add_scalar_row(
                        summary,
                        "soma_coverage",
                        case_id,
                        "expected_uncovered_coordinates",
                        empirical_uncovered,
                        theoretical_uncovered,
                        **parameters,
                    )
                    add_scalar_row(
                        summary,
                        "soma_coverage",
                        case_id,
                        "all_coordinates_covered_probability",
                        empirical_all_covered,
                        theoretical_all_covered,
                        **parameters,
                    )

                    records.append({
                        "case_id": case_id,
                        **parameters,
                        "coordinate_coverage_empirical": empirical_coordinate_coverage,
                        "coordinate_coverage_theoretical": one_coordinate_probability,
                        "expected_uncovered_empirical": empirical_uncovered,
                        "expected_uncovered_theoretical": theoretical_uncovered,
                        "all_covered_empirical": empirical_all_covered,
                        "all_covered_theoretical": theoretical_all_covered,
                    })

    return pd.DataFrame(records)


# =============================================================================
# Experiment 4: DE donor moments
# =============================================================================

def experiment_de_donor(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
    vectors: list[dict],
    matrices: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []

    population_sizes = [6, 10] if config.mode == "smoke" else [6, 10, 20, 50]
    dimensions = [3] if config.mode == "smoke" else [3, 10]
    weights = [0.5, 0.9] if config.mode == "smoke" else [0.2, 0.5, 0.8, 1.0]

    for population_size in population_sizes:
        for dimension in dimensions:
            population = make_correlated_population(
                population_size,
                dimension,
                rng,
            )
            target_indices = [0] if config.mode == "smoke" else [0, population_size // 2]

            for target_index in target_indices:
                for differential_weight in weights:
                    case_id = (
                        f"N{population_size}_D{dimension}_i{target_index}"
                        f"_F{differential_weight}"
                    )
                    theory = de_donor_theory(
                        population,
                        target_index,
                        differential_weight,
                    )

                    # Exact enumeration remains inexpensive for the selected sizes.
                    exact_mutants, exact_weights = enumerate_de_mutants(
                        population,
                        target_index,
                        differential_weight,
                    )
                    exact_mean, exact_covariance = weighted_mean_cov(
                        exact_mutants,
                        exact_weights,
                    )

                    mc_mutants = sample_de_mutants(
                        population,
                        target_index,
                        differential_weight,
                        config.mc_samples,
                        rng,
                    )
                    mc_mean, mc_covariance = empirical_mean_cov(mc_mutants)

                    theoretical_mean = np.asarray(theory["mean"])
                    theoretical_covariance = np.asarray(theory["covariance"])
                    parameters = {
                        "N": population_size,
                        "D": dimension,
                        "target_index": target_index,
                        "F": differential_weight,
                        "pool_size": int(theory["pool_size"]),
                        "n_mc": config.mc_samples,
                        "n_exact_tuples": len(exact_mutants),
                    }

                    add_scalar_row(
                        summary,
                        "de_donor",
                        case_id,
                        "exact_mean_scaled_l2_error",
                        vector_scaled_error(
                            exact_mean, theoretical_mean, theoretical_covariance
                        ),
                        None,
                        **parameters,
                    )
                    add_scalar_row(
                        summary,
                        "de_donor",
                        case_id,
                        "exact_covariance_relative_fro_error",
                        matrix_relative_error(
                            exact_covariance,
                            theoretical_covariance,
                        ),
                        None,
                        **parameters,
                    )
                    add_scalar_row(
                        summary,
                        "de_donor",
                        case_id,
                        "mc_mean_scaled_l2_error",
                        vector_scaled_error(
                            mc_mean, theoretical_mean, theoretical_covariance
                        ),
                        None,
                        **parameters,
                    )
                    add_scalar_row(
                        summary,
                        "de_donor",
                        case_id,
                        "mc_covariance_relative_fro_error",
                        matrix_relative_error(
                            mc_covariance,
                            theoretical_covariance,
                        ),
                        None,
                        **parameters,
                    )
                    add_scalar_row(
                        summary,
                        "de_donor",
                        case_id,
                        "covariance_multiplier",
                        float(
                            np.trace(exact_covariance)
                            / max(np.trace(np.asarray(theory["pool_covariance"])), EPS)
                        ),
                        float(theory["multiplier"]),
                        **parameters,
                    )

                    add_vector_entries(
                        vectors,
                        "de_donor",
                        case_id,
                        "mutant_mean_exact",
                        exact_mean,
                        theoretical_mean,
                        **parameters,
                    )
                    add_matrix_entries(
                        matrices,
                        "de_donor",
                        case_id,
                        "mutant_covariance_exact",
                        exact_covariance,
                        theoretical_covariance,
                        **parameters,
                    )
                    add_matrix_entries(
                        matrices,
                        "de_donor",
                        case_id,
                        "mutant_covariance_mc",
                        mc_covariance,
                        theoretical_covariance,
                        **parameters,
                    )

                    records.append({
                        "case_id": case_id,
                        **parameters,
                        "theoretical_multiplier": float(theory["multiplier"]),
                        "exact_mean_scaled_l2_error": vector_scaled_error(
                            exact_mean, theoretical_mean, theoretical_covariance
                        ),
                        "exact_covariance_relative_fro_error": matrix_relative_error(
                            exact_covariance,
                            theoretical_covariance,
                        ),
                        "mc_mean_scaled_l2_error": vector_scaled_error(
                            mc_mean, theoretical_mean, theoretical_covariance
                        ),
                        "mc_covariance_relative_fro_error": matrix_relative_error(
                            mc_covariance,
                            theoretical_covariance,
                        ),
                    })

    return pd.DataFrame(records)


# =============================================================================
# Experiment 5: exact forced DE crossover
# =============================================================================

def experiment_de_forced_crossover(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
    vectors: list[dict],
    matrices: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []

    dimensions = [2, 3] if config.mode == "smoke" else [2, 3, 5, 10]
    crossover_rates = [0.2, 0.8] if config.mode == "smoke" else [0.0, 0.1, 0.5, 0.9, 1.0]

    for dimension in dimensions:
        target = rng.normal(size=dimension)
        mutant = target + rng.normal(size=dimension)
        delta = mutant - target

        for crossover_rate in crossover_rates:
            case_id = f"D{dimension}_CR{crossover_rate}"
            theory = de_forced_conditional_theory(
                target,
                mutant,
                crossover_rate,
            )

            exact_masks, exact_weights = enumerate_forced_crossover_masks(
                dimension,
                crossover_rate,
            )
            exact_trials = target + exact_masks * delta
            exact_mean, exact_covariance = weighted_mean_cov(
                exact_trials,
                exact_weights,
            )

            mc_masks = sample_forced_crossover_masks(
                config.mc_samples,
                dimension,
                crossover_rate,
                rng,
            ).astype(float)
            mc_trials = target + mc_masks * delta
            mc_mean, mc_covariance = empirical_mean_cov(mc_trials)

            theoretical_mean = np.asarray(theory["mean"])
            theoretical_covariance = np.asarray(theory["covariance"])

            # Idealized independent/unforced Bernoulli model for comparison.
            independent_mean = target + crossover_rate * delta
            independent_covariance = (
                crossover_rate
                * (1.0 - crossover_rate)
                * np.diag(delta**2)
            )

            parameters = {
                "D": dimension,
                "CR": crossover_rate,
                "n_mc": config.mc_samples,
                "support_size": len(exact_masks),
            }

            add_scalar_row(
                summary,
                "de_forced_crossover",
                case_id,
                "effective_activation_probability",
                float(np.mean(exact_masks.T @ exact_weights)),
                float(theory["alpha"]),
                **parameters,
            )
            add_scalar_row(
                summary,
                "de_forced_crossover",
                case_id,
                "exact_mean_relative_l2_error",
                vector_relative_error(exact_mean, theoretical_mean),
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "de_forced_crossover",
                case_id,
                "exact_covariance_relative_fro_error",
                matrix_relative_error(
                    exact_covariance,
                    theoretical_covariance,
                ),
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "de_forced_crossover",
                case_id,
                "mc_covariance_relative_fro_error",
                matrix_relative_error(
                    mc_covariance,
                    theoretical_covariance,
                ),
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "de_forced_crossover",
                case_id,
                "forced_vs_unforced_mean_l2",
                float(np.linalg.norm(theoretical_mean - independent_mean)),
                None,
                **parameters,
            )
            add_scalar_row(
                summary,
                "de_forced_crossover",
                case_id,
                "forced_vs_unforced_covariance_fro",
                float(
                    np.linalg.norm(
                        theoretical_covariance - independent_covariance,
                        ord="fro",
                    )
                ),
                None,
                **parameters,
            )

            add_vector_entries(
                vectors,
                "de_forced_crossover",
                case_id,
                "trial_mean_exact",
                exact_mean,
                theoretical_mean,
                **parameters,
            )
            add_matrix_entries(
                matrices,
                "de_forced_crossover",
                case_id,
                "trial_covariance_exact",
                exact_covariance,
                theoretical_covariance,
                **parameters,
            )
            add_matrix_entries(
                matrices,
                "de_forced_crossover",
                case_id,
                "trial_covariance_mc",
                mc_covariance,
                theoretical_covariance,
                **parameters,
            )

            records.append({
                "case_id": case_id,
                **parameters,
                "effective_activation_probability": float(theory["alpha"]),
                "mask_offdiagonal_covariance": (
                    float(np.asarray(theory["gamma"])[0, 1])
                    if dimension > 1 else 0.0
                ),
                "exact_mean_relative_l2_error": vector_relative_error(
                    exact_mean,
                    theoretical_mean,
                ),
                "exact_covariance_relative_fro_error": matrix_relative_error(
                    exact_covariance,
                    theoretical_covariance,
                ),
                "mc_mean_relative_l2_error": vector_relative_error(
                    mc_mean,
                    theoretical_mean,
                ),
                "mc_covariance_relative_fro_error": matrix_relative_error(
                    mc_covariance,
                    theoretical_covariance,
                ),
                "forced_vs_unforced_mean_l2": float(
                    np.linalg.norm(theoretical_mean - independent_mean)
                ),
                "forced_vs_unforced_covariance_fro": float(
                    np.linalg.norm(
                        theoretical_covariance - independent_covariance,
                        ord="fro",
                    )
                ),
            })

    return pd.DataFrame(records)


# =============================================================================
# Experiment 6: total DE moments
# =============================================================================

def experiment_de_total(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
    vectors: list[dict],
    matrices: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []

    settings = (
        [(8, 3), (20, 10)]
        if config.mode == "smoke"
        else [(8, 3), (20, 5), (30, 10)]
    )
    differential_weights = [0.4, 0.9] if config.mode == "smoke" else [0.3, 0.6, 0.9]
    crossover_rates = [0.3, 0.9] if config.mode == "smoke" else [0.2, 0.5, 0.9]

    for population_size, dimension in settings:
        population = make_correlated_population(
            population_size,
            dimension,
            rng,
        )
        target_index = 0

        for differential_weight in differential_weights:
            for crossover_rate in crossover_rates:
                case_id = (
                    f"N{population_size}_D{dimension}"
                    f"_F{differential_weight}_CR{crossover_rate}"
                )
                theory = de_total_theory(
                    population,
                    target_index,
                    differential_weight,
                    crossover_rate,
                )

                mutant_accumulator = MomentAccumulator(dimension)
                trial_accumulator = MomentAccumulator(dimension)

                for start, end in iter_batches(
                    config.mc_samples,
                    config.batch_size,
                ):
                    batch = end - start
                    mutants, trials = sample_de_trials(
                        population,
                        target_index,
                        differential_weight,
                        crossover_rate,
                        batch,
                        rng,
                    )
                    mutant_accumulator.update(mutants)
                    trial_accumulator.update(trials)

                mutant_mean, mutant_covariance = mutant_accumulator.finalize()
                trial_mean, trial_covariance = trial_accumulator.finalize()

                theoretical_mutant_mean = np.asarray(theory["mutant_mean"])
                theoretical_mutant_covariance = np.asarray(theory["mutant_covariance"])
                theoretical_trial_mean = np.asarray(theory["trial_mean"])
                theoretical_trial_covariance = np.asarray(theory["trial_covariance"])

                parameters = {
                    "N": population_size,
                    "D": dimension,
                    "target_index": target_index,
                    "F": differential_weight,
                    "CR": crossover_rate,
                    "n_mc": config.mc_samples,
                }

                # Exact enumeration for low-dimensional cases. This combines
                # every valid ordered donor triple with every forced-crossover
                # mask and provides a machine-precision verification of the
                # total-moment formula.
                exact_trial_mean = None
                exact_trial_covariance = None
                n_exact_trials = 0
                if dimension <= 5:
                    exact_mutants, donor_weights = enumerate_de_mutants(
                        population, target_index, differential_weight
                    )
                    exact_masks, mask_weights = enumerate_forced_crossover_masks(
                        dimension, crossover_rate
                    )
                    target = population[target_index]
                    exact_trials = (
                        target[None, None, :]
                        + exact_masks[None, :, :]
                        * (exact_mutants[:, None, :] - target[None, None, :])
                    ).reshape(-1, dimension)
                    exact_weights = (
                        donor_weights[:, None] * mask_weights[None, :]
                    ).reshape(-1)
                    exact_trial_mean, exact_trial_covariance = weighted_mean_cov(
                        exact_trials, exact_weights
                    )
                    n_exact_trials = len(exact_trials)

                scalar_metrics = {
                    "mutant_mean_scaled_l2_error": vector_scaled_error(
                        mutant_mean,
                        theoretical_mutant_mean,
                        theoretical_mutant_covariance,
                    ),
                    "mutant_covariance_relative_fro_error": matrix_relative_error(
                        mutant_covariance,
                        theoretical_mutant_covariance,
                    ),
                    "trial_mean_scaled_l2_error": vector_scaled_error(
                        trial_mean,
                        theoretical_trial_mean,
                        theoretical_trial_covariance,
                    ),
                    "trial_covariance_relative_fro_error": matrix_relative_error(
                        trial_covariance,
                        theoretical_trial_covariance,
                    ),
                    "trial_covariance_trace": float(np.trace(trial_covariance)),
                    "trial_largest_eigenvalue": float(
                        np.linalg.eigvalsh(trial_covariance)[-1]
                    ),
                    "trial_condition_number": covariance_condition_number(
                        trial_covariance
                    ),
                    "mutant_trial_principal_angle_degrees": principal_angle_degrees(
                        mutant_covariance,
                        trial_covariance,
                    ),
                }

                if exact_trial_mean is not None and exact_trial_covariance is not None:
                    scalar_metrics.update({
                        "exact_trial_mean_scaled_l2_error": vector_scaled_error(
                            exact_trial_mean,
                            theoretical_trial_mean,
                            theoretical_trial_covariance,
                        ),
                        "exact_trial_covariance_relative_fro_error": (
                            matrix_relative_error(
                                exact_trial_covariance,
                                theoretical_trial_covariance,
                            )
                        ),
                    })

                for metric, value in scalar_metrics.items():
                    add_scalar_row(
                        summary,
                        "de_total",
                        case_id,
                        metric,
                        value,
                        None,
                        **parameters,
                    )

                add_vector_entries(
                    vectors,
                    "de_total",
                    case_id,
                    "mutant_mean_mc",
                    mutant_mean,
                    theoretical_mutant_mean,
                    **parameters,
                )
                add_vector_entries(
                    vectors,
                    "de_total",
                    case_id,
                    "trial_mean_mc",
                    trial_mean,
                    theoretical_trial_mean,
                    **parameters,
                )
                add_matrix_entries(
                    matrices,
                    "de_total",
                    case_id,
                    "mutant_covariance_mc",
                    mutant_covariance,
                    theoretical_mutant_covariance,
                    **parameters,
                )
                add_matrix_entries(
                    matrices,
                    "de_total",
                    case_id,
                    "trial_covariance_mc",
                    trial_covariance,
                    theoretical_trial_covariance,
                    **parameters,
                )
                if exact_trial_mean is not None and exact_trial_covariance is not None:
                    add_vector_entries(
                        vectors,
                        "de_total",
                        case_id,
                        "trial_mean_exact",
                        exact_trial_mean,
                        theoretical_trial_mean,
                        **parameters,
                    )
                    add_matrix_entries(
                        matrices,
                        "de_total",
                        case_id,
                        "trial_covariance_exact",
                        exact_trial_covariance,
                        theoretical_trial_covariance,
                        **parameters,
                    )

                records.append({
                    "case_id": case_id,
                    **parameters,
                    "n_exact_trials": n_exact_trials,
                    **scalar_metrics,
                    "mutant_covariance_trace": float(np.trace(mutant_covariance)),
                })

    return pd.DataFrame(records)


# =============================================================================
# Experiment 7: boundary repair
# =============================================================================

def experiment_boundary_repair(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []
    lower, upper = -1.0, 1.0

    # SOMA cases near the boundary.
    dimensions = [3] if config.mode == "smoke" else [3, 10]
    for dimension in dimensions:
        x = np.full(dimension, 0.85)
        leader = np.linspace(-0.85, 0.35, dimension)
        r = leader - x

        for p in ([0.5] if config.mode == "smoke" else [0.3, 0.8]):
            for t in ([2.2] if config.mode == "smoke" else [1.2, 2.2, 3.0]):
                masks = sample_bernoulli_masks(
                    config.boundary_samples,
                    dimension,
                    p,
                    rng,
                ).astype(float)
                raw = x + t * masks * r

                for repair_name, repair_function in (
                    ("clipping", clip_repair),
                    ("reflection", reflect_repair),
                ):
                    repaired = repair_function(raw, lower, upper)
                    metrics = boundary_distortion_metrics(
                        raw,
                        repaired,
                        lower,
                        upper,
                    )
                    case_id = f"SOMA_D{dimension}_p{p}_t{t}_{repair_name}"
                    parameters = {
                        "algorithm": "SOMA",
                        "D": dimension,
                        "p": p,
                        "t": t,
                        "F": np.nan,
                        "CR": np.nan,
                        "repair": repair_name,
                        "n_samples": config.boundary_samples,
                    }

                    for metric, value in metrics.items():
                        add_scalar_row(
                            summary,
                            "boundary_repair",
                            case_id,
                            metric,
                            value,
                            None,
                            **parameters,
                        )
                    records.append({"case_id": case_id, **parameters, **metrics})

    # DE cases using a population deliberately close to the box boundary.
    for dimension in dimensions:
        population_size = max(8, dimension + 4)
        population = rng.uniform(-0.95, 0.95, size=(population_size, dimension))
        population[0] = np.full(dimension, 0.85)
        target_index = 0

        for differential_weight in (
            [0.9] if config.mode == "smoke" else [0.5, 1.0]
        ):
            for crossover_rate in (
                [0.8] if config.mode == "smoke" else [0.3, 0.9]
            ):
                _, raw = sample_de_trials(
                    population,
                    target_index,
                    differential_weight,
                    crossover_rate,
                    config.boundary_samples,
                    rng,
                )

                for repair_name, repair_function in (
                    ("clipping", clip_repair),
                    ("reflection", reflect_repair),
                ):
                    repaired = repair_function(raw, lower, upper)
                    metrics = boundary_distortion_metrics(
                        raw,
                        repaired,
                        lower,
                        upper,
                    )
                    case_id = (
                        f"DE_D{dimension}_F{differential_weight}"
                        f"_CR{crossover_rate}_{repair_name}"
                    )
                    parameters = {
                        "algorithm": "DE",
                        "D": dimension,
                        "p": np.nan,
                        "t": np.nan,
                        "F": differential_weight,
                        "CR": crossover_rate,
                        "repair": repair_name,
                        "n_samples": config.boundary_samples,
                    }

                    for metric, value in metrics.items():
                        add_scalar_row(
                            summary,
                            "boundary_repair",
                            case_id,
                            metric,
                            value,
                            None,
                            **parameters,
                        )
                    records.append({"case_id": case_id, **parameters, **metrics})

    return pd.DataFrame(records)


# =============================================================================
# Experiment 8: selection distortion
# =============================================================================

def experiment_selection_distortion(
    config: RunConfig,
    rng: np.random.Generator,
    summary: list[dict],
) -> pd.DataFrame:
    records: list[dict] = []

    dimensions = [3] if config.mode == "smoke" else [3, 10]
    objective_names = ["sphere"] if config.mode == "smoke" else ["sphere", "quadratic"]

    for dimension in dimensions:
        quadratic = make_rotated_quadratic(dimension, rng)
        objectives = {
            "sphere": sphere_values,
            "quadratic": quadratic,
        }

        # ---------------------------------------------------------------------
        # SOMA path selection
        # ---------------------------------------------------------------------
        x = np.linspace(1.0, 1.8, dimension)
        leader = 0.25 * x
        displacement = leader - x
        path_parameters = np.linspace(0.2, 2.0, 10)

        for objective_name in objective_names:
            objective = objectives[objective_name]
            x_fitness = float(objective(x))

            for p in ([0.5] if config.mode == "smoke" else [0.3, 0.8]):
                raw_accumulator = MomentAccumulator(dimension)
                selected_accumulator = MomentAccumulator(dimension)
                selected_changed = 0
                improvement_sum = 0.0

                for start, end in iter_batches(
                    config.selection_trials,
                    config.batch_size,
                ):
                    batch = end - start
                    masks = (
                        rng.random(
                            (batch, len(path_parameters), dimension)
                        ) < p
                    )
                    proposals = (
                        x[None, None, :]
                        + path_parameters[None, :, None]
                        * masks
                        * displacement[None, None, :]
                    )

                    raw_accumulator.update(proposals.reshape(-1, dimension))

                    proposal_fitness = objective(proposals)
                    all_fitness = np.concatenate(
                        [
                            np.full((batch, 1), x_fitness),
                            proposal_fitness,
                        ],
                        axis=1,
                    )
                    best_index = np.argmin(all_fitness, axis=1)

                    selected = np.repeat(x[None, :], batch, axis=0)
                    changed = best_index > 0
                    selected[changed] = proposals[
                        np.arange(batch)[changed],
                        best_index[changed] - 1,
                    ]

                    selected_fitness = objective(selected)
                    selected_accumulator.update(selected)
                    selected_changed += int(changed.sum())
                    improvement_sum += float(
                        np.sum(x_fitness - selected_fitness)
                    )

                raw_mean, raw_covariance = raw_accumulator.finalize()
                selected_mean, selected_covariance = selected_accumulator.finalize()

                metrics = {
                    "selection_rate": selected_changed / config.selection_trials,
                    "mean_objective_improvement": (
                        improvement_sum / config.selection_trials
                    ),
                    "mean_shift_l2": float(
                        np.linalg.norm(selected_mean - raw_mean)
                    ),
                    "covariance_shift_fro": float(
                        np.linalg.norm(
                            selected_covariance - raw_covariance,
                            ord="fro",
                        )
                    ),
                    "raw_covariance_trace": float(np.trace(raw_covariance)),
                    "selected_covariance_trace": float(
                        np.trace(selected_covariance)
                    ),
                }

                case_id = f"SOMA_D{dimension}_{objective_name}_p{p}"
                parameters = {
                    "algorithm": "SOMA",
                    "D": dimension,
                    "objective": objective_name,
                    "p": p,
                    "F": np.nan,
                    "CR": np.nan,
                    "n_trials": config.selection_trials,
                }

                for metric, value in metrics.items():
                    add_scalar_row(
                        summary,
                        "selection_distortion",
                        case_id,
                        metric,
                        value,
                        None,
                        **parameters,
                    )
                records.append({"case_id": case_id, **parameters, **metrics})

        # ---------------------------------------------------------------------
        # DE greedy target-trial selection
        # ---------------------------------------------------------------------
        population_size = max(10, dimension + 5)
        population = make_correlated_population(
            population_size,
            dimension,
            rng,
            scale=0.8,
        )
        target_index = 0
        population[target_index] += np.linspace(1.0, 1.6, dimension)
        target = population[target_index]

        for objective_name in objective_names:
            objective = objectives[objective_name]
            target_fitness = float(objective(target))

            de_settings = (
                [(0.8, 0.8)]
                if config.mode == "smoke"
                else [(0.4, 0.3), (0.9, 0.9)]
            )

            for differential_weight, crossover_rate in de_settings:
                raw_accumulator = MomentAccumulator(dimension)
                selected_accumulator = MomentAccumulator(dimension)
                accepted = 0
                improvement_sum = 0.0

                for start, end in iter_batches(
                    config.selection_trials,
                    config.batch_size,
                ):
                    batch = end - start
                    _, trials = sample_de_trials(
                        population,
                        target_index,
                        differential_weight,
                        crossover_rate,
                        batch,
                        rng,
                    )
                    raw_accumulator.update(trials)

                    trial_fitness = objective(trials)
                    accept = trial_fitness <= target_fitness
                    selected = np.repeat(target[None, :], batch, axis=0)
                    selected[accept] = trials[accept]

                    selected_accumulator.update(selected)
                    accepted += int(accept.sum())
                    improvement_sum += float(
                        np.sum(target_fitness - objective(selected))
                    )

                raw_mean, raw_covariance = raw_accumulator.finalize()
                selected_mean, selected_covariance = selected_accumulator.finalize()

                metrics = {
                    "selection_rate": accepted / config.selection_trials,
                    "mean_objective_improvement": (
                        improvement_sum / config.selection_trials
                    ),
                    "mean_shift_l2": float(
                        np.linalg.norm(selected_mean - raw_mean)
                    ),
                    "covariance_shift_fro": float(
                        np.linalg.norm(
                            selected_covariance - raw_covariance,
                            ord="fro",
                        )
                    ),
                    "raw_covariance_trace": float(np.trace(raw_covariance)),
                    "selected_covariance_trace": float(
                        np.trace(selected_covariance)
                    ),
                }

                case_id = (
                    f"DE_D{dimension}_{objective_name}"
                    f"_F{differential_weight}_CR{crossover_rate}"
                )
                parameters = {
                    "algorithm": "DE",
                    "D": dimension,
                    "objective": objective_name,
                    "p": np.nan,
                    "F": differential_weight,
                    "CR": crossover_rate,
                    "n_trials": config.selection_trials,
                }

                for metric, value in metrics.items():
                    add_scalar_row(
                        summary,
                        "selection_distortion",
                        case_id,
                        metric,
                        value,
                        None,
                        **parameters,
                    )
                records.append({"case_id": case_id, **parameters, **metrics})

    return pd.DataFrame(records)


# =============================================================================
# Figures
# =============================================================================

def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(
        figures_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        figures_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def generate_figures(
    output_dir: Path,
    soma_independent: pd.DataFrame,
    soma_conditioned: pd.DataFrame,
    soma_coverage: pd.DataFrame,
    de_donor: pd.DataFrame,
    de_forced: pd.DataFrame,
    de_total: pd.DataFrame,
    boundary: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
    })

    # 1. Monte Carlo convergence for SOMA.
    convergence = soma_independent[
        soma_independent.get("is_convergence_case", False) == True
    ].copy()
    if not convergence.empty:
        fig, ax = plt.subplots(figsize=(6.3, 4.2))
        ax.loglog(
            convergence["n_samples"],
            convergence["mean_relative_l2_error"],
            marker="o",
            label="Mean error",
        )
        ax.loglog(
            convergence["n_samples"],
            convergence["covariance_relative_fro_error"],
            marker="s",
            label="Covariance error",
        )
        ax.set_xlabel("Monte Carlo samples")
        ax.set_ylabel("Relative error")
        ax.set_title("SOMA moment convergence")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figures_dir, "soma_mc_convergence")

    # 2. Conditioned SOMA negative mask covariance.
    if not soma_conditioned.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for dimension, group in soma_conditioned.groupby("D"):
            group = group.sort_values("p")
            ax.plot(
                group["p"],
                group["mask_offdiagonal_theoretical"],
                marker="o",
                label=fr"$D={dimension}$ theory",
            )
            ax.scatter(
                group["p"],
                group["mask_offdiagonal_empirical_mean"],
                marker="x",
                s=45,
            )
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlabel(r"Activation probability $p$")
        ax.set_ylabel("Off-diagonal mask covariance")
        ax.set_title("Dependence induced by rejecting all-zero SOMA masks")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figures_dir, "soma_conditioned_offdiagonal")

    # 3. Coordinate coverage.
    subset = soma_coverage[
        (soma_coverage["mask_mode"] == "independent")
        & (soma_coverage["D"] == soma_coverage["D"].max())
    ]
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for p, group in subset.groupby("p"):
            group = group.sort_values("J")
            ax.plot(
                group["J"],
                group["all_covered_theoretical"],
                label=fr"$p={p}$ theory",
            )
            ax.scatter(
                group["J"],
                group["all_covered_empirical"],
                s=24,
            )
        ax.set_xlabel("Number of independently masked path points $J$")
        ax.set_ylabel("Probability all coordinates are covered")
        ax.set_title(fr"Path-wide coverage, $D={int(subset['D'].iloc[0])}$")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figures_dir, "soma_path_coverage")

    # 4. DE donor covariance multiplier.
    if not de_donor.empty:
        representative = de_donor[
            (de_donor["D"] == de_donor["D"].min())
            & (de_donor["target_index"] == 0)
        ]
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for population_size, group in representative.groupby("N"):
            group = group.sort_values("F")
            ax.plot(
                group["F"],
                group["theoretical_multiplier"],
                marker="o",
                label=fr"$N={population_size}$",
            )
        ax.set_xlabel(r"Differential weight $F$")
        ax.set_ylabel(r"$\operatorname{tr}(\Sigma_v)/\operatorname{tr}(S_{-i})$")
        ax.set_title("Finite-population DE mutant covariance scale")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figures_dir, "de_donor_covariance_multiplier")

    # 5. Forced versus independent crossover.
    if not de_forced.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for dimension, group in de_forced.groupby("D"):
            group = group.sort_values("CR")
            ax.plot(
                group["CR"],
                group["forced_vs_unforced_covariance_fro"],
                marker="o",
                label=fr"$D={dimension}$",
            )
        ax.set_xlabel(r"Crossover rate $\mathrm{CR}$")
        ax.set_ylabel("Covariance difference (Frobenius norm)")
        ax.set_title("Standard forced DE/bin versus independent crossover")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figures_dir, "de_forced_vs_unforced")

    # 6. DE trial covariance trace across F and CR.
    if not de_total.empty:
        representative = de_total[
            (de_total["N"] == de_total["N"].max())
            & (de_total["D"] == de_total["D"].max())
        ]
        if representative.empty:
            representative = de_total
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for crossover_rate, group in representative.groupby("CR"):
            group = group.sort_values("F")
            ax.plot(
                group["F"],
                group["trial_covariance_trace"],
                marker="o",
                label=fr"$\mathrm{{CR}}={crossover_rate}$",
            )
        ax.set_xlabel(r"Differential weight $F$")
        ax.set_ylabel("Trial covariance trace")
        ax.set_title("Parameter control of DE trial spread")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figures_dir, "de_trial_covariance_trace")

    # 7. Boundary distortion.
    if not boundary.empty:
        grouped = (
            boundary.groupby(["algorithm", "repair"], as_index=False)
            ["covariance_shift_fro"].mean()
        )
        labels = [
            f"{algorithm}\n{repair}"
            for algorithm, repair in zip(grouped["algorithm"], grouped["repair"])
        ]
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        ax.bar(labels, grouped["covariance_shift_fro"])
        ax.set_ylabel("Mean covariance distortion (Frobenius norm)")
        ax.set_title("Boundary-repair distortion")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        save_figure(fig, figures_dir, "boundary_covariance_distortion")

    # 8. Selection distortion.
    if not selection.empty:
        grouped = (
            selection.groupby(["algorithm", "objective"], as_index=False)
            [["selection_rate", "covariance_shift_fro"]].mean()
        )
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for algorithm, group in grouped.groupby("algorithm"):
            ax.scatter(
                group["selection_rate"],
                group["covariance_shift_fro"],
                s=65,
                label=algorithm,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    row["objective"],
                    (row["selection_rate"], row["covariance_shift_fro"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_xlabel("Selection / acceptance rate")
        ax.set_ylabel("Pre/post-selection covariance shift")
        ax.set_title("Selection distorts the proposal distribution")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figures_dir, "selection_covariance_distortion")


# =============================================================================
# Reporting
# =============================================================================

def print_console_summary(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    print("\n" + "=" * 79)
    print("NUMERICAL EXPERIMENTS COMPLETE")
    print("=" * 79)
    print(f"Results directory: {output_dir.resolve()}")

    for name, dataframe in tables.items():
        print(f"\n{name}: {len(dataframe)} cases")
        error_columns = [
            column
            for column in dataframe.columns
            if "error" in column and ("relative" in column or "scaled" in column)
        ]
        for column in error_columns:
            finite = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            if len(finite):
                print(
                    f"  {column}: "
                    f"mean={finite.mean():.3e}, max={finite.max():.3e}"
                )

    print("\nUpload the generated ZIP archive in a future message.")
    print("It contains all CSV tables, figures, metadata, and diagnostics.")


def save_outputs(
    output_dir: Path,
    config: RunConfig,
    tables: dict[str, pd.DataFrame],
    summary_rows: list[dict],
    vector_rows: list[dict],
    matrix_rows: list[dict],
    elapsed_seconds: float,
) -> Path:
    ensure_dir(output_dir)

    for name, dataframe in tables.items():
        dataframe.to_csv(output_dir / f"{name}.csv", index=False)

    summary_dataframe = pd.DataFrame(summary_rows)
    vector_dataframe = pd.DataFrame(vector_rows)
    matrix_dataframe = pd.DataFrame(matrix_rows)

    summary_dataframe.to_csv(output_dir / "combined_summary.csv", index=False)
    vector_dataframe.to_csv(output_dir / "vector_entries.csv", index=False)
    matrix_dataframe.to_csv(output_dir / "matrix_entries.csv", index=False)

    report_lines = [
        "# Numerical experiment summary",
        "",
        f"- Mode: `{config.mode}`",
        f"- Seed: `{config.seed}`",
        f"- Elapsed time: `{elapsed_seconds:.3f}` seconds",
        "- Covariances use population normalization `1/n`.",
        "",
    ]
    for name, dataframe in tables.items():
        report_lines.extend([f"## {name}", "", f"Cases: **{len(dataframe)}**", ""])
        error_columns = [
            column for column in dataframe.columns
            if ("error" in column and ("relative" in column or "scaled" in column))
        ]
        if error_columns:
            report_lines.append("| Diagnostic | Mean | Maximum |")
            report_lines.append("|---|---:|---:|")
            for column in error_columns:
                values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
                if len(values):
                    report_lines.append(
                        f"| `{column}` | {values.mean():.6e} | {values.max():.6e} |"
                    )
            report_lines.append("")

    (output_dir / "summary_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    metadata = {
        "mode": config.mode,
        "seed": config.seed,
        "mc_samples": config.mc_samples,
        "coverage_trials": config.coverage_trials,
        "selection_trials": config.selection_trials,
        "boundary_samples": config.boundary_samples,
        "batch_size": config.batch_size,
        "pair_replicates": config.pair_replicates,
        "convergence_samples": list(config.convergence_samples),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "covariance_convention": (
            "All empirical and theoretical covariance matrices use population "
            "normalization 1/n rather than the unbiased 1/(n-1) estimator."
        ),
        "de_donor_sampling": (
            "Ordered, distinct donors sampled uniformly without replacement "
            "from the target-excluded donor pool."
        ),
        "de_crossover": (
            "Standard binomial crossover with one uniformly selected forced "
            "coordinate j_rand."
        ),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    completion_text = (
        "Run completed successfully.\n"
        f"Mode: {config.mode}\n"
        f"Seed: {config.seed}\n"
        f"Elapsed seconds: {elapsed_seconds:.3f}\n"
        f"Summary rows: {len(summary_dataframe)}\n"
        f"Vector rows: {len(vector_dataframe)}\n"
        f"Matrix rows: {len(matrix_dataframe)}\n"
    )
    (output_dir / "RUN_COMPLETE.txt").write_text(
        completion_text,
        encoding="utf-8",
    )

    archive_base = output_dir.parent / output_dir.name
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=output_dir.parent,
            base_dir=output_dir.name,
        )
    )
    return archive_path


# =============================================================================
# Main
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SOMA and DE stochastic-geometry experiments."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "quick", "full"),
        default="quick",
        help="Experiment size. Use full for final paper results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260319,
        help="Master NumPy random seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("operator_geometry_results"),
        help="Output directory.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Save tables only.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    config = build_config(arguments.mode, arguments.seed)
    output_dir = arguments.output

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    rng = np.random.default_rng(config.seed)
    summary_rows: list[dict] = []
    vector_rows: list[dict] = []
    matrix_rows: list[dict] = []

    start_time = time.perf_counter()

    print(f"Running mode={config.mode}, seed={config.seed}")
    print("1/8 SOMA independent-mask moments...")
    soma_independent = experiment_soma_independent(
        config, rng, summary_rows, vector_rows, matrix_rows
    )

    print("2/8 SOMA nonzero-conditioned masks...")
    soma_conditioned = experiment_soma_conditioned(
        config, rng, summary_rows, vector_rows, matrix_rows
    )

    print("3/8 SOMA path coverage...")
    soma_coverage = experiment_soma_coverage(
        config, rng, summary_rows
    )

    print("4/8 DE donor moments...")
    de_donor = experiment_de_donor(
        config, rng, summary_rows, vector_rows, matrix_rows
    )

    print("5/8 Forced-coordinate DE crossover...")
    de_forced = experiment_de_forced_crossover(
        config, rng, summary_rows, vector_rows, matrix_rows
    )

    print("6/8 Combined DE trial moments...")
    de_total = experiment_de_total(
        config, rng, summary_rows, vector_rows, matrix_rows
    )

    print("7/8 Boundary-repair distortion...")
    boundary = experiment_boundary_repair(
        config, rng, summary_rows
    )

    print("8/8 Selection distortion...")
    selection = experiment_selection_distortion(
        config, rng, summary_rows
    )

    tables = {
        "soma_independent": soma_independent,
        "soma_conditioned": soma_conditioned,
        "soma_coverage": soma_coverage,
        "de_donor": de_donor,
        "de_forced_crossover": de_forced,
        "de_total": de_total,
        "boundary_repair": boundary,
        "selection_distortion": selection,
    }

    if not arguments.skip_figures:
        print("Generating summary figures...")
        generate_figures(
            output_dir,
            soma_independent,
            soma_conditioned,
            soma_coverage,
            de_donor,
            de_forced,
            de_total,
            boundary,
            selection,
        )

    elapsed = time.perf_counter() - start_time
    archive_path = save_outputs(
        output_dir,
        config,
        tables,
        summary_rows,
        vector_rows,
        matrix_rows,
        elapsed,
    )
    print_console_summary(output_dir, tables)
    print(f"\nZIP archive: {archive_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())