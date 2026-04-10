#!/usr/bin/env python3
"""Compute per-day R_tree and Forman-Ricci metrics from PostgreSQL.

For each dataset, this script reads events directly from PostgreSQL, builds one
undirected simple graph per day, computes:

1. R_tree = (|V| - number_of_connected_components) / |E|
2. Mean unweighted 1D Forman-Ricci edge curvature:
   F(u, v) = 4 - deg(u) - deg(v)

It then reports both the per-day values and the average across the selected days.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from statistics import mean

import networkx as nx
import psycopg2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pidsmaker.config.config import DATASET_DEFAULT_CONFIG
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import datetime_to_ns_time_US

DEFAULT_DATASETS = ["CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3", "optc_h051"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-day R_tree and Forman-Ricci metrics from PostgreSQL."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process. Default: CADETS_E3 THEIA_E3 CLEARSCOPE_E3 optc_h051",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["train", "val", "test"],
        help="Dataset splits whose days should be evaluated. Default: train val test",
    )
    parser.add_argument("--database-host", default="postgres", help="PostgreSQL host.")
    parser.add_argument("--database-port", default="5432", help="PostgreSQL port.")
    parser.add_argument("--database-user", default="postgres", help="PostgreSQL user.")
    parser.add_argument("--database-password", default="postgres", help="PostgreSQL password.")
    parser.add_argument(
        "--use-all-files",
        action="store_true",
        help="Use dataset.database_all_file instead of dataset.database.",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=50000,
        help="Server-side cursor fetch size. Default: 50000",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional dataset-average CSV output path.",
    )
    parser.add_argument(
        "--per-day-csv-out",
        default="",
        help="Optional per-day CSV output path.",
    )
    return parser.parse_args()


def resolve_days(dataset: str, splits: list[str]) -> list[str]:
    cfg = DATASET_DEFAULT_CONFIG[dataset]
    if "all" in splits:
        splits = ["train", "val", "test"]

    days = []
    for split in splits:
        days.extend(cfg[f"{split}_files"])
    return days


def day_bounds_ns(dataset: str, day_name: str) -> tuple[int, int]:
    cfg = DATASET_DEFAULT_CONFIG[dataset]
    day_num = int(day_name.split("_")[-1])
    start = f"{cfg['year_month']}-{day_num} 00:00:00"
    stop = f"{cfg['year_month']}-{day_num + 1} 00:00:00"
    return datetime_to_ns_time_US(start), datetime_to_ns_time_US(stop)


def get_database_name(dataset: str, use_all_files: bool) -> str:
    cfg = DATASET_DEFAULT_CONFIG[dataset]
    return cfg["database_all_file"] if use_all_files else cfg["database"]


def connect_to_database(args: argparse.Namespace, dataset: str):
    return psycopg2.connect(
        database=get_database_name(dataset, args.use_all_files),
        host=args.database_host,
        user=args.database_user,
        password=args.database_password,
        port=args.database_port,
    )


def _build_dataset_cfg(dataset: str):
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.dataset = _Cfg()
    cfg.dataset.name = dataset
    return cfg


def get_include_operations(dataset: str) -> list[str]:
    ops = list(get_rel2id(_build_dataset_cfg(dataset)).keys())
    return [op for op in ops if isinstance(op, str)]


def infer_split(dataset: str, day: str) -> str:
    cfg = DATASET_DEFAULT_CONFIG[dataset]
    for split in ["train", "val", "test", "unused"]:
        if day in cfg[f"{split}_files"]:
            return split
    return "unknown"


def make_time_filter_for_days(dataset: str, day_names: list[str]) -> tuple[str, list[int]]:
    clauses = []
    params: list[int] = []
    for day in day_names:
        start_ns, end_ns = day_bounds_ns(dataset, day)
        clauses.append("(timestamp_rec > %s AND timestamp_rec < %s)")
        params.extend([start_ns, end_ns])
    return " OR ".join(clauses), params


def normalize_undirected_edge(src: str, dst: str) -> tuple[str, str]:
    try:
        src_key = int(src)
        dst_key = int(dst)
        return (src, dst) if src_key <= dst_key else (dst, src)
    except ValueError:
        return tuple(sorted((src, dst)))


def compute_r_tree(graph: nx.Graph) -> float:
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    if num_nodes <= 1 or num_edges == 0:
        return 0.0
    return (num_nodes - nx.number_connected_components(graph)) / num_edges


def compute_forman_ricci_mean(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    degrees = dict(graph.degree())
    values = [4.0 - degrees[u] - degrees[v] for u, v in graph.edges()]
    return mean(values)


def fetch_nodes(conn, dataset: str, day_names: list[str], fetch_size: int) -> set[str]:
    time_filter, time_params = make_time_filter_for_days(dataset, day_names)
    include_ops = get_include_operations(dataset)
    cursor = conn.cursor(name=f"nodes_{dataset.lower()}")
    cursor.itersize = fetch_size
    cursor.execute(
        f"""
        WITH filtered AS (
            SELECT src_index_id::text AS src_index_id, dst_index_id::text AS dst_index_id
            FROM event_table
            WHERE ({time_filter})
              AND operation = ANY(%s)
        )
        SELECT node
        FROM (
            SELECT DISTINCT src_index_id AS node FROM filtered
            UNION
            SELECT DISTINCT dst_index_id AS node FROM filtered
        ) dedup_nodes
        ORDER BY node;
        """,
        [*time_params, include_ops],
    )

    nodes: set[str] = set()
    while True:
        rows = cursor.fetchmany(fetch_size)
        if not rows:
            break
        for (node,) in rows:
            nodes.add(str(node))
    cursor.close()
    return nodes


def fetch_edges(
    conn,
    dataset: str,
    day_names: list[str],
    fetch_size: int,
) -> set[tuple[str, str]]:
    time_filter, time_params = make_time_filter_for_days(dataset, day_names)
    include_ops = get_include_operations(dataset)
    cursor = conn.cursor(name=f"edges_{dataset.lower()}")
    cursor.itersize = fetch_size
    cursor.execute(
        f"""
        SELECT DISTINCT
            CASE
                WHEN src_index_id::bigint <= dst_index_id::bigint THEN src_index_id::text
                ELSE dst_index_id::text
            END AS src,
            CASE
                WHEN src_index_id::bigint <= dst_index_id::bigint THEN dst_index_id::text
                ELSE src_index_id::text
            END AS dst
        FROM event_table
        WHERE ({time_filter})
          AND operation = ANY(%s)
          AND src_index_id <> dst_index_id
        ORDER BY src, dst;
        """,
        [*time_params, include_ops],
    )

    edges: set[tuple[str, str]] = set()
    while True:
        rows = cursor.fetchmany(fetch_size)
        if not rows:
            break
        for src, dst in rows:
            edges.add(normalize_undirected_edge(str(src), str(dst)))
    cursor.close()
    return edges


def build_graph(nodes: set[str], edges: set[tuple[str, str]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def compute_day_metrics(conn, dataset: str, day: str, fetch_size: int) -> dict:
    split = infer_split(dataset, day)
    print(f"[{dataset}] {day}: querying distinct nodes...", flush=True)
    nodes = fetch_nodes(conn, dataset, [day], fetch_size)
    print(f"[{dataset}] {day}: querying distinct undirected edges...", flush=True)
    edges = fetch_edges(conn, dataset, [day], fetch_size)

    graph = build_graph(nodes, edges)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_components = nx.number_connected_components(graph) if num_nodes > 0 else 0

    return {
        "dataset": dataset,
        "day": day,
        "split": split,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_components": num_components,
        "R_tree": compute_r_tree(graph),
        "Forman_Ricci_mean": compute_forman_ricci_mean(graph),
    }


def compute_dataset_metrics(dataset: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    conn = connect_to_database(args, dataset)
    try:
        day_rows = []
        for day in resolve_days(dataset, args.splits):
            day_rows.append(compute_day_metrics(conn, dataset, day, args.fetch_size))
    finally:
        conn.close()

    avg_row = {
        "dataset": dataset,
        "database": get_database_name(dataset, args.use_all_files),
        "splits": ",".join(args.splits),
        "num_days": len(day_rows),
        "avg_num_nodes": mean(row["num_nodes"] for row in day_rows),
        "avg_num_edges": mean(row["num_edges"] for row in day_rows),
        "avg_num_components": mean(row["num_components"] for row in day_rows),
        "R_tree_mean": mean(row["R_tree"] for row in day_rows),
        "Forman_Ricci_mean_avg": mean(row["Forman_Ricci_mean"] for row in day_rows),
    }
    return day_rows, avg_row


def write_csv(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    per_day_rows = []
    avg_rows = []

    for dataset in args.datasets:
        if dataset not in DATASET_DEFAULT_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset}")
        day_rows, avg_row = compute_dataset_metrics(dataset, args)
        per_day_rows.extend(day_rows)
        avg_rows.append(avg_row)

    width = max(len(row["dataset"]) for row in avg_rows) + 2
    print("Per-day metrics")
    print(
        f"{'Dataset'.ljust(width)}{'Day':>10} {'Split':>8} {'Nodes':>12} {'Edges':>12} "
        f"{'R_tree':>12} {'Forman_Ricci_mean':>20}"
    )
    for row in per_day_rows:
        print(
            f"{row['dataset'].ljust(width)}{row['day']:>10} {row['split']:>8} "
            f"{row['num_nodes']:>12d} {row['num_edges']:>12d} {row['R_tree']:>12.6f} "
            f"{row['Forman_Ricci_mean']:>20.6f}"
        )

    print("\nDataset averages")
    print(
        f"{'Dataset'.ljust(width)}{'Days':>8} {'AvgNodes':>12} {'AvgEdges':>12} "
        f"{'R_tree_mean':>14} {'Forman_Ricci_mean_avg':>24}"
    )
    for row in avg_rows:
        print(
            f"{row['dataset'].ljust(width)}{row['num_days']:>8d} "
            f"{row['avg_num_nodes']:>12.2f} {row['avg_num_edges']:>12.2f} "
            f"{row['R_tree_mean']:>14.6f} {row['Forman_Ricci_mean_avg']:>24.6f}"
        )

    if args.csv_out:
        write_csv(avg_rows, args.csv_out)
        print(f"\nSaved dataset averages to {args.csv_out}")

    if args.per_day_csv_out:
        write_csv(per_day_rows, args.per_day_csv_out)
        print(f"Saved per-day metrics to {args.per_day_csv_out}")


if __name__ == "__main__":
    main()
