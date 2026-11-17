# Pool-based active learning for tabular foundation models

Implementation of the paper *Active In-Context Learning for Tabular Foundation Models* (Treerath & Pittorino, 2026)

https://arxiv.org/abs/2603.27385



#### In-context learning
The tabular foundation model (TFM) is **TabPFN** by default.
TabPFN learns in context, labelling a sample **only extends its context**, without weight update.

#### Standard learning
Any scikit-learn classifier with `predict_proba` can be used instead.
It is refit from scratch after each labelled batch, like CatBoost and XGBoost baselines in the paper.


## Acquisition strategies
- `"random"`: uniform sampling, passive baseline.
- `"margin"`: queries the samples with the smallest gap between the two most probable classes.
- `"coreset"`: greedy k-center. Repeatedly selects the sample farthest from the labelled
  set in the preprocessed feature space.
- `"hybrid"`: keeps the most entropic half of the pool (> `2 * batch_size`
  samples), runs k-means with k = `batch_size` then queries the sample nearest to each centroid.
- `"proxy_hybrid"`: balanced logistic regression proxy. It's fit on the labelled set,
  shortlists the most entropic 5 % of the pool (clamp between 200 & 2000).
  <!-- The classifier then scores only this shortlist, keeps its `3 * batch_size` most
  entropic samples and applies the same k-means step. This cuts classifier calls
  by roughly 20x on large pools. -->

## Preprocessing
The preprocessing is **fit once** on the pool **before any query**.

- Numerical features are mean-imputed & normalised.
- Categorical features are mode-imputed & ordinal-encoded.
- Integer features with fewer than 20 unique values are treated as categorical.


## Usage
Unlabelled samples are marked with `-1`, `NaN` or `None` in `y`, as in `sklearn.semi_supervised`.
The paper starts with one labelled sample per class.

### Getting started
```python
n = 100  # labelling budget

learner = ActiveLearner(strategy="hybrid", batch_size=10, random_state=0)
learner.fit(X_pool, y_partial)
while learner.n_labeled_ < n:
    indices = learner.query()                      # rows of X_pool to label
    learner.label(indices, ask_expert(indices))    # labels given by the expert
learner.predict_proba(X_new)
```
