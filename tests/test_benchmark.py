import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tab_aicl.benchmark import aulc_table, compare, main, run_benchmark, split, summary_table


def synthetic_loader(name):
    X, y = make_classification(400, n_features=5, n_informative=3, random_state=len(name))
    X = pd.DataFrame(X).assign(group=np.where(X[:, 0] > 0, "high", "low"))
    return X, y, [5]


def test_split_subsamples_then_stratifies():
    X, y, _ = synthetic_loader("abc")
    X_pool, X_test, y_pool, y_test = split(X, y, seed=0, max_samples=100)
    assert (len(X_pool), len(X_test)) == (70, 30)
    assert abs(y_pool.mean() - y_test.mean()) < 0.1


def test_run_benchmark_resumes(tmp_path):
    output = tmp_path / "results.csv"
    arguments = dict(
        datasets=["one", "two"], methods=["LabelSpreading-Random"], seeds=range(2), budget=30, output=output, loader=synthetic_loader, verbose=False
    )
    results = run_benchmark(**arguments)
    assert set(results["dataset"]) == {"one", "two"}
    assert len(aulc_table(results)) == 4

    def loader_that_must_not_run(name):
        raise AssertionError("finished runs must be skipped")

    resumed = run_benchmark(**{**arguments, "loader": loader_that_must_not_run})
    assert len(resumed) == len(results) == len(pd.read_csv(output))


def test_run_benchmark_rejects_unknown_methods():
    with pytest.raises(ValueError, match="Unknown methods"):
        run_benchmark(["one"], ["TabPFN-BALD"], loader=synthetic_loader)


def fake_results(n_seeds=10):
    rows = []
    for dataset, gap in (("easy", 0.2), ("tie", 0.0)):
        for seed in range(n_seeds):
            for method, shift in (("TabPFN-Hybrid", gap), ("CatBoost-Margin", 0.0)):
                for n_labeled, kappa in ((2, 0.3), (12, 0.5), (22, 0.6)):
                    noise = 0.01 * ((seed * 7) % 5)
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "batch_size": 10,
                            "seed": seed,
                            "n_labeled": n_labeled,
                            "kappa": kappa + shift + noise,
                            "roc_auc": 0.8,
                        }
                    )
    return pd.DataFrame(rows)


def test_summary_and_comparison():
    results = fake_results()
    table = summary_table(results)
    assert list(table.columns) == ["TabPFN-Hybrid", "CatBoost-Margin"]
    assert list(table.index) == ["easy", "tie", "Average"]
    assert table.loc["tie", "TabPFN-Hybrid"] == table.loc["tie", "CatBoost-Margin"]
    # (0.5 + 0.7) / 2 * 10 + (0.7 + 0.8) / 2 * 10 = 13.5 over 20 labels, plus a mean noise of 0.02
    assert table.loc["easy", "TabPFN-Hybrid"].startswith("0.695")

    final = summary_table(results, final=True)
    assert final.loc["tie", "CatBoost-Margin"].startswith("0.6")

    comparison = compare(results, "TabPFN-Hybrid", "CatBoost-Margin")
    assert comparison.loc["easy", "result"] == "Significantly higher"
    assert comparison.loc["easy", "delta_aulc"] == pytest.approx(0.2)
    assert comparison.loc["tie", "result"] == "No significant difference"
    assert comparison.loc["tie", "p_value"] == 1.0


def test_summarize_command(tmp_path, capsys):
    path = tmp_path / "results.csv"
    fake_results().to_csv(path, index=False)
    main(["summarize", str(path), "--compare", "TabPFN-Hybrid", "CatBoost-Margin", "--plot", "easy", "--plot-dir", str(tmp_path / "figures")])
    printed = capsys.readouterr().out
    assert "Significantly higher" in printed
    assert (tmp_path / "figures" / "easy.png").exists()
