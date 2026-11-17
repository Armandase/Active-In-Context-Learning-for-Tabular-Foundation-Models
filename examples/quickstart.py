"""Active learning with TabPFN on the breast cancer dataset.

The first part is the labelling loop of a real project, the expert being
simulated by the true labels. The second part compares the acquisition
strategies with the protocol of the paper.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from tab_aicl import STRATEGIES, ActiveLearner, initial_context, normalized_aulc, run_active_learning

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
X_pool, y_pool = X_pool.reset_index(drop=True), y_pool.to_numpy()

# 1. Labelling loop: start from one labelled sample per class, -1 marks the unlabelled ones
y_partial = np.full(len(y_pool), -1)
start = initial_context(y_pool, random_state=0)
y_partial[start] = y_pool[start]

learner = ActiveLearner(strategy="hybrid", batch_size=10, random_state=0).fit(X_pool, y_partial)
while learner.n_labeled_ < 50:
    indices = learner.query()  # positions in X_pool of the samples to label
    learner.label(indices, y_pool[indices])  # an expert would label X_pool.iloc[indices] here
    kappa = cohen_kappa_score(y_test, learner.predict(X_test))
    print(f"{learner.n_labeled_:>3} labels  kappa {kappa:.3f}")

# 2. Strategies compared with the simulated protocol of the paper
print("\nNormalized AULC of Cohen's kappa up to 50 labels")
for strategy in STRATEGIES:
    curve = run_active_learning(X_pool, y_pool, X_test, y_test, strategy=strategy, budget=50, random_state=0)
    print(f"{strategy:<13} {normalized_aulc(curve['n_labeled'], curve['kappa']):.3f}")
