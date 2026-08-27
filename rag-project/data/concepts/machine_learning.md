# Machine Learning — GATE DA reference notes

## Bias, variance and overfitting
Expected prediction error decomposes into bias^2 + variance + irreducible
noise. A high-bias model underfits (too simple to capture the signal); a
high-variance model overfits (it fits noise in the training set and
generalizes poorly). Increasing model complexity lowers bias and raises
variance, which is why validation error is U-shaped in complexity while
training error decreases monotonically.

Regularization (L2 / ridge, L1 / lasso) constrains parameters to reduce
variance at the cost of some bias. L1 drives coefficients exactly to zero and
therefore performs feature selection; L2 shrinks them smoothly but rarely to
zero.

## Cross validation
k-fold cross validation splits the data into k parts, trains on k-1 and
validates on the held-out fold, repeating k times and averaging. It gives a
lower-variance estimate of generalization error than a single split.
Leave-one-out is the k = n extreme: nearly unbiased but high variance and
expensive. Any preprocessing fitted on data (scaling, PCA, feature selection)
must be fitted *inside* each fold, otherwise information leaks from validation
into training and the estimate is optimistic.

## Classification metrics
For a binary confusion matrix with TP, FP, TN, FN:
- Precision = TP / (TP + FP): of what we flagged, how much was right.
- Recall (sensitivity) = TP / (TP + FN): of what was truly positive, how much
  we caught.
- F1 = harmonic mean of precision and recall = 2PR / (P + R).
- Accuracy = (TP + TN) / total, which is misleading on imbalanced data.

The ROC curve plots TPR against FPR across thresholds; AUC is the probability
that a randomly chosen positive is ranked above a randomly chosen negative.
A random classifier has AUC 0.5.

## Decision trees
Splits are chosen to maximize purity gain. Entropy of a node is
H = -sum p_i * log2(p_i); information gain is the parent entropy minus the
weighted average child entropy. Gini impurity is 1 - sum p_i^2 and behaves
similarly. A fully grown tree can always reach zero training error on
non-contradictory data, which is precisely why it overfits and why pruning or
depth limits are needed.

## Naive Bayes
Applies Bayes' theorem with the "naive" assumption that features are
conditionally independent given the class:
P(C|x) proportional to P(C) * product over j of P(x_j | C).
The independence assumption is usually false, yet the classifier is often
competitive because the argmax over classes can be correct even when the
probability estimates themselves are poorly calibrated.

## k-nearest neighbours
No training phase; prediction is a majority vote among the k closest training
points. k = 1 gives zero training error but high variance. Increasing k
smooths the decision boundary, raising bias and lowering variance. kNN
requires feature scaling, because an unscaled large-range feature dominates
the distance metric.

## Support vector machines
An SVM finds the hyperplane maximizing the margin to the nearest points, the
support vectors. Only the support vectors determine the boundary; moving other
points does not change it. The soft-margin parameter C trades margin width
against misclassification: large C penalizes errors heavily and can overfit.
The kernel trick replaces inner products with a kernel function, allowing a
linear separator in an implicit higher-dimensional space without ever forming
the coordinates.

## k-means clustering
Alternates assigning points to the nearest centroid and recomputing centroids
as cluster means. It monotonically decreases within-cluster sum of squares and
therefore converges, but only to a local optimum that depends on
initialization — hence k-means++ seeding and multiple restarts. It assumes
roughly spherical, similarly sized clusters.
