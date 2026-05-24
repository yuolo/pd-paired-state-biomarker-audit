"""Graph utilities for the connectomic prior."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np


def adjacency_to_graph(matrix: np.ndarray, region_names: list[str]) -> nx.Graph:
    """Convert an adjacency matrix into a weighted NetworkX graph."""

    graph = nx.Graph()
    graph.add_nodes_from(region_names)
    for i, left in enumerate(region_names):
        for j, right in enumerate(region_names):
            if j <= i:
                continue
            weight = float(matrix[i, j])
            if weight > 0:
                graph.add_edge(left, right, weight=weight)
    return graph


def graph_summary(matrix: np.ndarray, region_names: list[str]) -> dict[str, float | int]:
    """Compute basic graph statistics for documentation and QC."""

    graph = adjacency_to_graph(matrix, region_names)
    degrees = dict(graph.degree(weight="weight"))
    return {
        "n_regions": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": float(nx.density(graph)),
        "mean_weighted_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "is_connected": int(nx.is_connected(graph)) if graph.number_of_nodes() else 0,
    }


def laplacian_spectrum(matrix: np.ndarray, k: int | None = None) -> np.ndarray:
    """Return sorted Laplacian eigenvalues for graph-shape comparisons."""

    degree = np.diag(matrix.sum(axis=1))
    laplacian = degree - matrix
    values = np.sort(np.linalg.eigvalsh(laplacian))
    if k is not None:
        return values[:k]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a saved normative graph.")
    parser.add_argument("--matrix", default="data/processed/normative_graph.npy")
    args = parser.parse_args()

    matrix = np.load(Path(args.matrix))
    names = [f"region_{idx}" for idx in range(matrix.shape[0])]
    print(graph_summary(matrix, names))


if __name__ == "__main__":
    main()
