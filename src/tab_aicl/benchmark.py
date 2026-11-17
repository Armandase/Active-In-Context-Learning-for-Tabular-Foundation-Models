"""Benchmark protocol of the paper (Section IV) on the OpenML datasets of Table I.

python -m tab_aicl.benchmark run --datasets iris vehicle --seeds 10
python -m tab_aicl.benchmark summarize results/results.csv --compare TabPFN-Hybrid CatBoost-Margin
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .baselines import ESTIMATORS, METHODS
from .simulation import normalized_aulc, run_active_learning

# name -> OpenML ID (Table I)
PAPER_DATASETS = {
    "iris": 61,
    "glass": 41,
    "ionosphere": 59,
    "balance-scale": 997,
    "vehicle": 994,
    "page-blocks": 1021,
    "parkinsons": 1488,
    "seeds": 1499,
    "bank-marketing": 45065,
    "adult": 45068,
    "covertype": 1596,
    "kc1": 1066,
    "jm1": 46979,
    "blood-transfusion": 46913,
    "diabetes": 46921,
    "tic-tac-toe": 137,
    "credit-g": 46918,
    "steel-plates": 46959,
    "phoneme": 44127,
    "ilpd": 41943,
}

KEY = ["dataset", "method", "batch_size", "seed"]


def load_openml(name: str) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Features, target and categorical columns (from the OpenML metadata) of a dataset.

    ``name`` is a key of ``PAPER_DATASETS`` or an OpenML ID.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=PAPER_DATASETS.get(name) or int(name), as_frame=True)
    X, y = data.data, data.target
    keep = y.notna().to_numpy()
    X, y = X[keep].reset_index(drop=True), np.asarray(y[keep])
    categorical = [
        position for position, dtype in enumerate(X.dtypes) if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
    ]
    return X, y, categorical


def split(X: pd.DataFrame, y: np.ndarray, seed: int, test_size: float = 0.3, max_samples: int = 10_000):
    """Uniform subsample to ``max_samples`` rows, then stratified pool/test split, both drawn from ``seed``."""
    if len(y) > max_samples:
        keep = np.sort(np.random.RandomState(seed).choice(len(y), max_samples, replace=False))
        X, y = X.iloc[keep], y[keep]
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool, y_test


def run_benchmark(
    datasets: Iterable[str] = tuple(PAPER_DATASETS),
    methods: Iterable[str] = tuple(METHODS),
    seeds: Iterable[int] = range(10),
    batch_size: int = 10,
    budget: int = 100,
    output: str | Path | None = None,
    loader: Callable[[str], tuple[pd.DataFrame, np.ndarray, list[int]]] = load_openml,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every method on every dataset and seed, as in the paper.

    For a given seed all methods share the split and the initial context.
    With ``output``, each finished run is appended to this CSV file and the
    runs already in it are skipped, so an interrupted benchmark resumes.
    Returns all the learning curves, one row per round.
    """
    methods, seeds = list(methods), list(seeds)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}, choose among {list(METHODS)}.")
    output = None if output is None else Path(output)
    previous = pd.read_csv(output) if output is not None and output.exists() else pd.DataFrame(columns=KEY)
    done = set(previous[KEY].drop_duplicates().astype(str).itertuples(index=False, name=None))
    curves = [previous]

    for dataset in datasets:
        pending = [(seed, method) for seed in seeds for method in methods if (dataset, method, str(batch_size), str(seed)) not in done]
        if not pending:
            continue
        X, y, categorical = loader(dataset)
        for seed in sorted({seed for seed, _ in pending}):
            X_pool, X_test, y_pool, y_test = split(X, y, seed)
            for method in [method for s, method in pending if s == seed]:
                estimator, strategy = METHODS[method]
                start = time.perf_counter()
                curve = run_active_learning(
                    X_pool,
                    y_pool,
                    X_test,
                    y_test,
                    strategy=strategy,
                    estimator=ESTIMATORS[estimator](seed),
                    batch_size=batch_size,
                    budget=budget,
                    categorical_features=categorical,
                    random_state=seed,
                )
                for column, value in zip(KEY, (dataset, method, batch_size, seed), strict=True):
                    curve.insert(KEY.index(column), column, value)
                curves.append(curve)
                if output is not None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    curve.to_csv(output, mode="a", header=not output.exists(), index=False)
                if verbose:
                    aulc = normalized_aulc(curve["n_labeled"], curve["kappa"])
                    print(f"{dataset:<18} {method:<22} seed {seed:<3} AULC {aulc:.3f}  ({time.perf_counter() - start:.0f}s)", flush=True)
    curves = [curve for curve in curves if len(curve)]
    return pd.concat(curves, ignore_index=True) if curves else previous


def aulc_table(results: pd.DataFrame, metric: str = "kappa") -> pd.DataFrame:
    """Normalized AULC of ``metric`` for each run (dataset, method, batch size, seed)."""
    return (
        results.sort_values("n_labeled")
        .groupby(KEY)
        .apply(lambda run: normalized_aulc(run["n_labeled"], run[metric]), include_groups=False)
        .rename("aulc")
        .reset_index()
    )


def summary_table(results: pd.DataFrame, metric: str = "kappa", final: bool = False) -> pd.DataFrame:
    """Mean ± std across seeds of the AULC (Table II), or of the final score with ``final`` (Tables V, VI).

    One row per dataset plus their average, one column per method.
    """
    if final:
        last = results.sort_values("n_labeled").groupby(KEY).tail(1)
        scores = last[KEY + [metric]].rename(columns={metric: "aulc"})
    else:
        scores = aulc_table(results, metric)
    stats = scores.groupby(["dataset", "method"])["aulc"].agg(["mean", "std"])
    table = stats.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1).unstack("method")
    table.loc["Average"] = stats["mean"].unstack("method").mean().map("{:.3f}".format)
    methods = [method for method in METHODS if method in table.columns]
    return table[methods + [method for method in table.columns if method not in methods]]


def compare(results: pd.DataFrame, method: str, baseline: str, metric: str = "kappa", alpha: float = 0.05) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test on the per-seed AULC of two methods (Table III).

    The p-values are Benjamini-Hochberg corrected across datasets.
    """
    from scipy.stats import false_discovery_control, wilcoxon

    scores = aulc_table(results, metric).pivot_table(index=["dataset", "seed"], columns="method", values="aulc")
    paired = scores[[method, baseline]].dropna()
    rows = []
    for dataset, group in paired.groupby(level="dataset"):
        difference = group[method] - group[baseline]
        p_value = wilcoxon(group[method], group[baseline]).pvalue if (difference != 0).any() else 1.0
        rows.append({"dataset": dataset, "delta_aulc": difference.mean(), "p_value": p_value})
    table = pd.DataFrame(rows).set_index("dataset")
    table["p_adjusted"] = false_discovery_control(table["p_value"].to_numpy(), method="bh")
    significant = table["p_adjusted"] < alpha
    table["result"] = np.where(
        ~significant, "No significant difference", np.where(table["delta_aulc"] > 0, "Significantly higher", "Significantly lower")
    )
    return table


def plot_learning_curves(results: pd.DataFrame, dataset: str, path: str | Path | None = None):
    """Mean learning curves with 95% confidence intervals across seeds, for Cohen's kappa and ROC AUC (Figs. 2-5)."""
    import matplotlib.pyplot as plt

    runs = results[results["dataset"] == dataset]
    order = {method: position for position, method in enumerate(METHODS)}
    curves = sorted(runs.groupby(["method", "batch_size"]), key=lambda item: (order.get(item[0][0], len(order)), item[0]))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric, title in zip(axes, ("kappa", "roc_auc"), ("Cohen's kappa", "ROC AUC"), strict=True):
        for (method, batch_size), run in curves:
            stats = run.groupby("n_labeled")[metric].agg(["mean", "sem"])
            half_width = 1.96 * stats["sem"].fillna(0)
            label = method if runs["batch_size"].nunique() == 1 else f"{method} (B={batch_size})"
            axis.plot(stats.index, stats["mean"], label=label)
            axis.fill_between(stats.index, stats["mean"] - half_width, stats["mean"] + half_width, alpha=0.15)
        axis.set(title=f"{dataset} - {title} (95% CI)", xlabel="Labeled samples", ylabel=title)
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize="small")
    if path is not None:
        figure.savefig(path, dpi=150)
    return figure


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tab_aicl.benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the active learning benchmark")
    run.add_argument("--datasets", nargs="+", default=list(PAPER_DATASETS), help="names of Table I or OpenML IDs (default: the 20 of the paper)")
    run.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS), metavar="METHOD", help=f"among {', '.join(METHODS)} (default: all)"
    )
    run.add_argument("--seeds", type=int, default=10, help="number of seeds, 0..N-1 (default: 10)")
    run.add_argument("--batch-size", type=int, default=10)
    run.add_argument("--budget", type=int, default=100)
    run.add_argument("--output", default="results/results.csv", help="CSV of the learning curves, resumed if it exists")

    summarize = commands.add_parser("summarize", help="print the tables of the paper from a results CSV")
    summarize.add_argument("results")
    summarize.add_argument("--metric", default="kappa", choices=["kappa", "roc_auc"])
    summarize.add_argument("--compare", nargs=2, metavar=("METHOD", "BASELINE"), help="Wilcoxon test with Benjamini-Hochberg correction")
    summarize.add_argument("--plot", nargs="*", metavar="DATASET", help="save the learning curves of these datasets (all if none given)")
    summarize.add_argument("--plot-dir", default="results/figures")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_benchmark(args.datasets, args.methods, range(args.seeds), args.batch_size, args.budget, args.output)
        return

    results = pd.read_csv(args.results)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for batch_size, group in results.groupby("batch_size"):
            print(f"\nNormalized AULC of {args.metric} (mean ± std across seeds), batch size {batch_size}\n")
            print(summary_table(group, args.metric).to_string())
            print(f"\nFinal {args.metric} (mean ± std across seeds), batch size {batch_size}\n")
            print(summary_table(group, args.metric, final=True).to_string())
            if args.compare:
                print(f"\n{args.compare[0]} vs {args.compare[1]}, batch size {batch_size}\n")
                print(compare(group, *args.compare, metric=args.metric).to_string(float_format="{:.3f}".format))
    if args.plot is not None:
        directory = Path(args.plot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for dataset in args.plot or sorted(results["dataset"].unique()):
            plot_learning_curves(results, dataset, directory / f"{dataset}.png")
            print(f"Saved {directory / f'{dataset}.png'}")


if __name__ == "__main__":
    main()
