# Q1 notes

## Shared setup (also used by Q2-Q4)
- Data: data/cifar10.npz built by common/prepare_data.py. Split: common/split_seed242003.npz,
  45000 train / 5000 val, stratified (4500/500 per class), seed 242003 from roll number M24MA2003.
- Normalisation: per-channel mean/std computed on the 45k train split only (not val, not test).
- Whole dataset sits on the GPU, batches drawn with a seeded generator. Augmentation (when used) is
  random horizontal flip + random 32x32 crop from the 4px zero-padded image, done on the GPU.
- Train accuracy in all curves is measured in eval mode on the un-augmented train images, so train and
  val numbers are comparable.

## Q1(a) design
Reason tested: the MLP has no spatial prior. It treats the 3072 inputs as an unordered vector, so it
has to learn locality and shift invariance from 45k images, which it cannot do well. It ends up fitting
the training set but generalising badly.

Competing explanation: lack of capacity / the MLP is just too small or too hard to optimise
(underfitting).

How the runs separate the two:
- If capacity were the issue, the MLP would have low TRAIN accuracy, and giving it 4x more parameters
  would raise val accuracy a lot.
- If the missing spatial prior is the issue, the MLP fits the train set (high train acc), val stays low,
  extra width barely helps, and a CNN with the same parameter count and the same training budget does
  much better. Giving the MLP flip/crop augmentation adds some of the invariance through the data
  instead, so it should help val a little and reduce overfitting, but not close the gap.

Fairness criterion (fixed before running):
- Matched parameter count: MLP 3072-336-336-10 has 1,150,474 params, CNN has 1,148,874 (0.14% apart).
  The wide MLP is deliberately 4.25x bigger (4,884,490) because it is the capacity test.
- Same training budget: 40 epochs over the same 45k images, batch 128, SGD (momentum 0.9, nesterov),
  OneCycle schedule, weight decay 5e-4, same seed.
- Same tuning effort: each model gets the same lr grid {0.01, 0.03, 0.1, 0.3}, chosen by final-epoch val
  accuracy. 0.3 was added after the first MLP runs picked 0.1 (the edge of the first grid), so that no
  model is picked at the boundary of the grid. All models were run on the full 4-value grid.
- Both models use BatchNorm + ReLU, so neither gets a normalisation advantage.
- Not matched: compute per step. The CNN does far more multiply-adds per image than the MLP for the same
  parameter count (weights are reused at every position). This is part of what a spatial prior means,
  so we match parameters and epochs, not FLOPs, and say so in the report.
- Test set: evaluated once per model, after the lr was chosen on val (q1a_select_test.py).

## Q1(b) design
- 20% of the 45k training labels replaced by a different class (uniform over the other 9), seed 242003.
  These cannot generalise: nothing in the image predicts the new label.
- Model: MLP 3072-2048-2048-10, 10.5M params (about 230 params per training image), BN, no weight decay,
  no augmentation, SGD momentum 0.9, constant lr 0.002, 60 epochs. Constant lr so that any change in
  the curves is not caused by the lr schedule.
- Pilots (logs kept in logs/): at lr 0.01 val peaked at epoch 5 and the train fit (on noisy labels)
  passed 0.90 by epoch 17. I expected a smaller lr to slow this down, but at lr 0.002 val still peaked at
  epoch ~5 and train fit reached 1.00 by epoch 30 (it memorised slightly faster: less SGD noise).
  So the problem was the resolution of the measurement, not the speed of training. Final runs
  measure 8 times per epoch (every ~44 steps) plus once before training, and 60 epochs is enough
  because everything is fitted by epoch ~30.
- Measured every epoch: fit on correctly labelled train images, fit of the wrong labels on relabelled
  images (memorisation), how often relabelled images are still predicted as their real class, val
  accuracy.
- Controls: 0% and 40% relabelled with the same model and settings.
- The best-val checkpoint (early stopping on val, which is allowed) and the final checkpoint are both
  evaluated on test after training finishes.

## Q1(a) results (selection on val, single test evaluation; Q1/results/q1a_summary.json)
lr grid was extended twice so no model is selected at the edge: first 0.3 was added (MLPs picked 0.1),
then 1.0 was added for every model (the CNN picked 0.3 by 0.900 vs 0.899). All models were run on the full
5-value grid {0.01, 0.03, 0.1, 0.3, 1.0}.

| model                | params    | lr  | train acc | val acc | test acc |
|----------------------|-----------|-----|-----------|---------|----------|
| MLP 336-336          | 1,150,474 | 0.3 | 0.971     | 0.578   | 0.573    |
| Wide MLP 1152-1152   | 4,884,490 | 0.1 | 1.000     | 0.600   | 0.587    |
| MLP + flip/crop      | 1,150,474 | 0.1 | 0.664     | 0.615   | 0.621    |
| CNN                  | 1,148,874 | 0.3 | 1.000     | 0.900   | 0.897    |

Interpretation:
- Underfitting/capacity is not the reason. The MLP fits 97% of the training set (the wide one 100%), so it
  can represent the training data. Giving it 4.25x more parameters adds only 1.4 test points (0.573 -> 0.587).
  The train-val gap (0.39 for the MLP) is what is large.
- The same parameter budget, same 40 epochs and same tuning give the CNN 0.897, +32 test points. The only thing
  that changed is how the parameters are wired (local 3x3 filters shared over positions, pooling).
- Flip/crop augmentation gives the MLP some of the invariance through the data. It shrinks the gap a lot
  (train 0.66 vs val 0.61) and improves test by 4.9 points, but the MLP is still 28 points behind the CNN.
  So invariance to small shifts is part of the story, but the main thing the MLP lacks is the locality/weight
  sharing prior itself, which it cannot learn from 45k images.
- Caveats: one seed per setting, so differences of 1-2 points between MLP variants are within normal
  run-to-run noise; the 30-point MLP vs CNN gap is not. Parameters and epochs are matched, FLOPs are not.

## Q1(b) results (Q1/results/q1b_noise*.json, table q1b_table_noise20.csv, figure q1b_memorisation)
20% relabelled run, selected epochs:

| epoch | train (correct labels) | relabelled -> true class | relabelled -> wrong label | val   |
|-------|------------------------|--------------------------|---------------------------|-------|
| 0     | 0.108                  | 0.114                    | 0.094                     | 0.113 |
| 1     | 0.534                  | 0.461                    | 0.086                     | 0.471 |
| 3     | 0.652                  | 0.471                    | 0.132                     | 0.497 |
| 4.125 | 0.710                  | 0.439                    | 0.190                     | 0.512 (best) |
| 8     | 0.818                  | 0.284                    | 0.431                     | 0.479 |
| 15    | 0.952                  | 0.088                    | 0.837                     | 0.459 |
| 30    | 1.000                  | 0.000                    | 1.000                     | 0.478 |

| relabelled | best val (epoch) | crossover epoch* | test, best-val ckpt | test, final |
|------------|------------------|------------------|---------------------|-------------|
| 0%         | 0.585 (28.75)    | -                | 0.565               | 0.564       |
| 20%        | 0.512 (4.1)      | 6.5              | 0.503               | 0.457       |
| 40%        | 0.467 (3.1)      | 5.9              | 0.459               | 0.395       |
*first measurement where more relabelled images are predicted as their wrong label than as their true class.

Interpretation:
- Yes, useful structure comes first. In the first ~3 epochs the network gets 47% of the relabelled training
  images "wrong" in the right way: it predicts their real class, even though it is being trained towards a
  different label. It can only do that by using features that generalise, which is also when val accuracy rises.
- Memorisation starts after that: from about epoch 3-4 the fraction of wrong labels fitted rises quickly
  (0.13 -> 0.43 by epoch 8 -> 1.00 by epoch 30), the "true class" fraction collapses, and val accuracy falls
  from its peak (0.512 at epoch 4.1) to about 0.46-0.48. The val peak sits before the crossover (6.5).
- With clean labels (0%) val keeps creeping up to epoch ~29 and does not drop, so the drop in the noisy runs
  comes from fitting the wrong labels, not from the long training itself. More noise (40%) gives an earlier peak,
  an earlier crossover and a bigger drop.
- On test, the val-selected early checkpoint is 4.6 points better than the final memorising model at 20% noise
  (0.503 vs 0.457) and 6.3 points at 40%, while at 0% the two are the same.
- The short dip around epoch 58 in all curves is a loss spike of the constant-lr, no-weight-decay run (a single
  unstable step that is recovered within an epoch). It is after everything else and does not affect the argument.

## Data source note (checked 23 Nov 02:40)
The data come from the Hugging Face parquet release uoft-cs/cifar10 (rev 0b27149...). Same images as the official
batches: per-channel mean/std over the 50k training images match the official values exactly
(0.4914/0.4822/0.4465, 0.2470/0.2435/0.2616) and the first test labels match the official test batch. But the
training images are in a different order (first labels 0,6,0,2,... instead of the official 6,9,9,4,...). The
split indices refer to the parquet order, so the README pins that revision and its checksums. The official file
could not be downloaded for a byte-level check (server at ~350 B/s, no resume support). Log: logs/data_check.log.
