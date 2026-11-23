# Q4 notes

## Fixed representation
- The 1024-d bottleneck code of the Q3(b) denoising autoencoder (encoder frozen, reused from Q3(b)).
- Why this one: among the Q3 encoders it had the highest validation linear-probe accuracy (0.553), so it
  is chosen on validation only. It was trained without labels, so clustering it in (c) is a genuinely
  unsupervised exercise (a supervised CNN feature would already have the labels built into it).
- Features standardised per dimension with the mean/std of the 45k train codes. Saved once in
  Q4/results/features.npz and not changed afterwards.

## SVM subset
- Fitting kernel SVMs on 45k x 1024 is slow (a single RBF fit on 10k takes about 2 minutes), so a stratified
  subset of 10,000 training images (1,000 per class), seed 242003, was drawn and saved to
  Q4/results/svm_subset_idx.npy before any SVM was fitted. Every SVM in the sweep uses it.
  Validation is the full 5k val split, test the full 10k test set.

## Sweep
- C in {0.001, 0.01, 0.1, 1, 10, 100} (five orders of magnitude). 0.001 was added after the linear kernel
  picked 0.01, the edge of the first grid {0.01,...,100}, kernels: linear, RBF (gamma='scale'),
  polynomial degree 3 (gamma='scale', coef0=1). sklearn SVC (libsvm), one-vs-one for 10 classes.
- Recorded per setting: train accuracy on the 10k subset, val accuracy, fraction of training points that are
  support vectors (in any of the 45 pairwise classifiers).
- Selection: highest val accuracy, then one refit and a single test evaluation.

## Clustering
- Candidates: k-means (k-means++, 10 inits) and diagonal GMM (3 inits), k in {5, 10, 15, 20, 30}, all fitted on
  the 45k train codes without labels. Chosen by validation ARI (val labels used only for selection, like
  the SVM). Validation NMI and silhouette are also reported.
- The final model is the selected one fitted on train; test codes are assigned with predict(). Test labels
  are only used after that, for ARI, NMI and the contingency table.

## (a) Sweep results (Q4/results/svm_sweep.json; train = 10k subset, val = 5k)
| kernel | C     | train  | val    | SV frac |
|--------|-------|--------|--------|---------|
| linear | 0.001 | 0.573  | 0.456  | 0.905   |
| linear | 0.01  | 0.681  | 0.467  | 0.853   |
| linear | 0.1   | 0.832  | 0.438  | 0.823   |
| linear | 1     | 0.954  | 0.422  | 0.807   |
| linear | 10    | 0.997  | 0.409  | 0.801   |
| linear | 100   | 1.000  | 0.406  | 0.799   |
| rbf    | 0.001 | 0.324  | 0.295  | 1.000   |
| rbf    | 0.01  | 0.324  | 0.295  | 1.000   |
| rbf    | 0.1   | 0.477  | 0.405  | 0.989   |
| rbf    | 1     | 0.841  | 0.512  | 0.953   |
| rbf    | 10    | 0.998  | 0.539  | 0.965   |
| rbf    | 100   | 1.000  | 0.539  | 0.964   |
| poly3  | 0.001 | 0.359  | 0.340  | 1.000   |
| poly3  | 0.01  | 0.489  | 0.429  | 0.988   |
| poly3  | 0.1   | 0.753  | 0.509  | 0.942   |
| poly3  | 1     | 0.974  | 0.531  | 0.928   |
| poly3  | 10    | 1.000  | 0.518  | 0.921   |
| poly3  | 100   | 1.000  | 0.516  | 0.920   |
Selected on val: RBF, C = 10 (val 0.5394; C = 100 gives 0.5392, a tie, the smaller C is kept by the max rule).

Interpretation:
- Small C = strong regularisation. RBF and poly at C <= 0.01 underfit badly (train ~0.32-0.49, val close to
  train) and every training point is a support vector: with a tiny C every point sits inside the margin with
  its dual variable at the bound C, so the solution is close to a weighted vote of all training points.
- Linear kernel: the classes are not linearly separable at small C, but with 1024 dimensions and 10k points it
  can reach 100 % train accuracy by C = 100. Val falls monotonically from C = 0.01 (0.467 -> 0.406): this is
  overfitting, more C buys train accuracy that does not transfer.
- RBF: val rises with C up to 10 and then stays flat; once train accuracy is ~100 % (C >= 10) the solution
  hardly changes (same SV fraction, same val), so larger C cannot overfit further. Poly peaks earlier (C = 1)
  and then loses 1.5 points, so it does overfit a little.
- The SV fraction stays very high everywhere (0.80-1.00). Classes overlap heavily in this representation, so
  most points are on or inside some pairwise margin. The linear kernel needs the fewest SVs but generalises worst;
  SV fraction alone is not a good predictor of val accuracy here. A drop in SV fraction with C (linear 0.905 ->
  0.799, rbf 1.0 -> 0.953) marks the move away from the underfitting regime.
- Fit times 1.5-5 min per setting on one CPU core each.

## (a) Test
Selected RBF C=10 refit on the same 10k subset, evaluated once: test accuracy 0.528 (val 0.539).
Q4/results/svm_test.json has the confusion matrix; the model is pickled in svm_selected.pkl.

## (b) Hard pairs (Q4/results/svm_pairs.json, figures q4b_pair_*.png)
Hard pairs = largest symmetric test confusion count of the 10-class SVM.
| pair             | test confusions | pairwise acc (ovo clf on the two classes) | separation ratio | pair SV frac |
|------------------|-----------------|-------------------------------------------|------------------|--------------|
| cat-dog          | 392             | 0.634                                     | 0.0148           | 0.947        |
| automobile-truck | 341             | 0.771                                     | 0.0270           | 0.923        |
| bird-deer        | 288             | 0.703                                     | 0.0168           | 0.838        |
| (easiest: frog-ship) | 28          | 0.947                                     | 0.165            |              |
separation ratio = squared distance between the two class means / average within-class spread (mean squared
distance to own mean), on the standardised training codes. Median over the 45 pairs: 0.064.
Spearman correlation between separation ratio and test confusions over all 45 pairs: -0.83.
- For each pair: 4 training SVs of each class closest to the pair boundary (they sit at |f| <= 1, i.e. on or
  inside the margin), and the 8 test images of the two classes with the smallest |f|.
- 6/8 (cat-dog), 4/8 (car-truck), 6/8 (bird-deer) of the nearest test images are misclassified.
- In this code the class means of the hard pairs are 2-4x closer (relative to spread) than a typical pair, and
  almost every training point of those classes is a support vector of the pair classifier: the boundary has
  to be supported by nearly the whole pair, i.e. the classes overlap rather than being separated by a gap.

## (c) Clustering (Q4/results/cluster_select.json, cluster_test.json, svm_vs_cluster.json)
Selection on validation (models fitted on the 45k train codes without labels):
| method | k  | val ARI | val NMI | val silhouette |
|--------|----|---------|---------|----------------|
| kmeans | 5  | 0.045   | 0.077   | 0.009          |
| kmeans | 10 | 0.062   | 0.112   | -0.009         |
| kmeans | 15 | 0.057   | 0.121   | -0.036         |
| kmeans | 20 | 0.049   | 0.126   | -0.042         |
| kmeans | 30 | 0.044   | 0.131   | -0.046         |
| gmm    | 5  | 0.051   | 0.096   | -0.085         |
| gmm    | 10 | 0.053   | 0.111   | -0.105         |
| gmm    | 15 | 0.046   | 0.111   | -0.112         |
| gmm    | 20 | 0.043   | 0.118   | -0.121         |
| gmm    | 30 | 0.040   | 0.126   | -0.123         |
Selected: k-means, k = 10 (best val ARI). NMI keeps growing with k (it rewards finer partitions), which is why
ARI was used for the choice. Silhouettes near 0 or negative: there is no clear cluster structure in this code.
Test (fit on train, test codes assigned with predict, labels used only afterwards): ARI 0.065, NMI 0.116.
Contingency table: figures/q4_contingency.png.

What the clusters follow: colour/background more than class. Cluster 1 = airplane 329 + ship 232 (blue sky or
water), cluster 8 = ship 366 + airplane 156 + truck 191, cluster 5 = bird 366 + deer 376 + frog 373 + horse 162
(green/brown natural backgrounds), cluster 2 = truck 270 + horse 246.

## Classification vs clustering disagreement
- automobile: the SVM's best class (test recall 0.668, highest of the 10), but the least coherent class under
  clustering together with cat: its largest cluster only holds 15.9% of the automobile test images and the
  row is spread over all 10 clusters (25-159 per cluster). Classification says "automobile is the most
  separable class in this representation", clustering says "automobile has no structure of its own".
- Pair level: automobile vs cat. The pairwise SVM separates them with 0.905 accuracy on their test images;
  a cluster-based vote (each cluster labelled by the majority of the two classes among its test members)
  reaches only 0.631, and 85% of the automobile+cat test images sit in clusters where both classes make up at
  least 25%. Same pattern for automobile vs horse (0.923 vs 0.658) and automobile vs dog (0.937 vs 0.675).
- Why: cars come in every colour and background, so the colour-dominated clusters split them, but within the
  code there are still directions (shape/edge related) that separate them from animals, and the RBF SVM finds
  them with labels. A class can be separable without being compact.
- No pair goes the other way (clustering above the SVM); the closest is cat vs dog (SVM 0.634, clusters 0.584),
  both near chance.
