# Pool-based active learning for tabular foundation models

Unofficial implementation of the paper *Active In-Context Learning for Tabular Foundation Models* (Treerath & Pittorino, 2026), [arXiv:2603.27385](https://arxiv.org/abs/2603.27385).

This repository follows the paper as closely as allows.
Check this section to understand [differences from the paper](#differences-from-the-paper).

## Resume

Tab-AICL (Tabular Active In-Context Learning) treats active learning as **context optimisation**.</br>
The predictor is a fixed tabular foundation model and each labelled sample is added to the context without any weight update.

#### In-context learning
The tabular foundation model (TFM) is **TabPFN** by default (`n_estimators=32`, as in the paper).</br>
TabPFN learns in context, labelling a sample **only extends its context**, 

#### Standard learning
Any scikit-learn classifier with `predict_proba` can be used instead.</br>
It is refit from scratch after each labelled batch, like the CatBoost and XGBoost baselines in the paper.</br>


## Installation
```bash
git clone https://github.com/Armandase/Active-In-Context-Learning-for-Tabular-Foundation-Models.git
cd Active-In-Context-Learning-for-Tabular-Foundation-Models
pip install -e .                # TabPFN & the active learner
pip install -e ".[benchmark]"   # CatBoost, XGBoost, SciPy and Matplotlib to reproduce paper's benchmark
```
The paper uses `tabpfn` 6.2.0 (`pip install tabpfn==6.2.0` to match it exactly).


## Usage
Unlabelled samples are marked with `-1`, `NaN` or `None` in `y`, as in `sklearn.semi_supervised`. </br>
The paper starts with one labelled sample per class (`initial_context` draws them).

### Getting started
```python
from tab_aicl import ActiveLearner

n = 100  # labelling budget

learner = ActiveLearner(strategy="hybrid", batch_size=10, random_state=0)
learner.fit(X_pool, y_partial)
while learner.n_labeled_ < n:
    # indices in the X_pool of the samples to label
    indices = learner.query()
    # labels given by the expert
    learner.label(indices, ask_expert(indices))
learner.predict_proba(X_new)
```
Any classifier can replace TabPFN, e.g. `ActiveLearner(CatBoostClassifier(verbose=0), strategy="margin")`.</br>

[examples/quickstart.py](examples/quickstart.py) runs the loop on the breast cancer dataset to compares strategies.


## Active learning strategies
| `strategy` | Paper | Rule |
|---|---|---|
| `"random"` | Random | Uniform sampling, passive baseline. |
| `"margin"` | Margin (Alg. 1) | Queries the samples with the smallest gap between the two most probable classes. |
| `"coreset"` | Coreset (Alg. 4) | Greedy k-center: repeatedly selects the sample farthest from the labelled set in the preprocessed feature space. |
| `"hybrid"` | Hybrid (Alg. 2) | Keeps the most entropic half of the pool (at least `2 * batch_size` samples), runs k-means with k = `batch_size`, then queries the sample nearest to each centroid. |
| `"proxy_hybrid"` | Proxy-Hybrid (Alg. 3) | A balanced logistic regression proxy, fit on the labelled set, shortlists the 5 % most entropic samples of the pool (clamped between 200 and 2000). TabPFN then scores only this shortlist, keeps its `3 * batch_size` most entropic samples and applies the same k-means step. It reduice TabPFN calls by up to 20x on large pools. |

Strategy classes are also parametrable, like `ActiveLearner(strategy=ProxyHybridSampling(filter_ratio=0.1))`.

##### Add a new strategy

A new rule subclasses `QueryStrategy`:
```python
from tab_aicl import QueryStrategy, entropy

class EntropySampling(QueryStrategy):
    def select(self, estimator, X_labeled, y_labeled, X_pool, batch_size, random_state):
        return entropy(estimator.predict_proba(X_pool)).argsort()[::-1][:batch_size]
```


## Differences from the paper
- **Early stopping of the GBDT baselines.** The paper sets `early_stopping_rounds=50` for CatBoost and XGBoost but does not say which validation set it monitors; none is used here, so they train for the full 2000 iterations.
- **Label Spreading** predicts uniform probabilities, instead of scikit-learn's NaN, for test samples whose neighbours are reached by no label.
- **Subsampling** of datasets above 10 000 rows is redrawn for each seed, with the split.
- **Categorical features** in the benchmark come from the OpenML metadata; the integer-valued rule is only the fallback, as described in the paper. TabPFN receives the preprocessed matrix, without categorical indices.
- **Unspecified details**: k-means uses `n_init=10`, entropy uses ε = 1e-12, and each refit of TabPFN draws its seed from the learner's `random_state`.
- **Outside the paper's protocol**, which always starts with one label per class: the first query without labels is random, and a single labelled class falls back to coreset sampling.
- **Known gap: TabPFN-Coreset on Ionosphere** reaches an AULC of 0.416 against 0.617 in the paper. Greedy k-center (Algorithm 4) fills the context with about 78 % of class `b` (36 % of the pool): the scattered "bad" radar returns are always the farthest points. The gap stays the same with the integer-valued categorical rule and without standardisation, so its cause is not found yet.


## Tests
```bash
pip install -e ".[benchmark,dev]"
pytest              # fast tests, with a logistic regression in place of TabPFN
pytest -m slow      # end-to-end rounds with the real TabPFN model
```

## License
The code of this repository is released under the [MIT License](LICENSE). The method and the paper belong to their authors.
