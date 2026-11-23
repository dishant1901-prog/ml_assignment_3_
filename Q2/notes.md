# Q2 notes

## Pre-registration (written 2026-11-22 22:31. No test number existed yet; the only CNN numbers seen were
the first 6 epochs of validation accuracy from the crashed Q1(a) CNN run, up to 0.743.)

### Q2(a) fairness criterion
The CNN and MLP are compared at a matched parameter count and a matched training budget:
- MLP 3072-336-336-10 (1,150,474 params) and the 6-layer CNN (1,148,874 params). These are the same two
  models as Q1(a) (reused, first introduced in Q1(a)).
- Same data (45k train split), 40 epochs, batch 128, SGD momentum 0.9 nesterov, OneCycle, weight decay 5e-4,
  same seed, BN+ReLU in both, no augmentation for either model (augmentation like crops and flips only
  makes sense for images with intact spatial layout, so it would not be the same treatment under the
  pixel permutation in (b)).
- Equal tuning effort: lr picked from the same grid on the validation split by final-epoch val accuracy.
- Compute per step is not matched (the CNN reuses its weights at every position, so it does more
  multiply-adds). We treat that as part of the architecture, not as an unfair extra.
- Test accuracy is computed once per model after selection.

Proposed reason for the gap (stated before results): the CNN's advantage comes from exploiting the local
spatial arrangement of pixels (nearby pixels are correlated, the same local pattern can appear anywhere),
via local receptive fields and weight sharing. The MLP cannot use this arrangement at all.

### Q2(b) prediction
Transformation: one fixed random permutation of the 1024 pixel positions (seed 242003), applied to all three
colour channels in the same way, to train, val and test images. Pixel values are unchanged, only their
positions move, and the mapping is the same for every image, so all information is still there.

Expected result:
- MLP: essentially no change (within about 1 point). A fully connected first layer has no notion of pixel
  order, so a fixed permutation of its inputs is equivalent to permuting its weight rows. The only
  difference is the random initialisation / batch order.
- CNN: a large drop, down to about the MLP level or below, because its 3x3 filters and pooling now see
  pixels that are not neighbours in the original image.
If the CNN drops to around the MLP level while the MLP does not change, the reason in (a) is supported.
If the CNN keeps most of its advantage, the reason in (a) is wrong (the gap would then have to come from
something like optimisation or normalisation instead).

### Q2(c) plan
Two explanations of what the CNN uses:
- (A) local evidence: textures, colours and small parts inside small windows ("bag of local features").
- (B) global layout: where parts sit relative to each other (object shape / configuration).
Experiment: split each 32x32 image into a grid of patches and shuffle the patches with a new random order
for every image, so local content inside each patch is intact but the global arrangement is destroyed.
Patch size controls how much local structure survives. Train and test on the same transformation.
If (A) is right, accuracy stays fairly high once patches are big enough to hold a recognisable part.
If (B) matters, accuracy drops a lot even with big patches. Errors (per-class, confusions) and
confidence (mean max-softmax for right and wrong predictions, calibration) are analysed on test.

## Results (Q2/results/q2_summary.json, q2_test_probs.npz; all lrs chosen on val)
| condition      | MLP lr | MLP val | MLP test | CNN lr | CNN val | CNN test |
|----------------|--------|---------|----------|--------|---------|----------|
| clean (= Q1a)  | 0.3    | 0.578   | 0.573    | 0.3    | 0.900   | 0.897    |
| pixel perm.    | 0.3    | 0.580   | 0.576    | 0.3    | 0.559   | 0.561    |
| 16x16 patches  | 0.3    | 0.400   | 0.403    | 0.3*   | 0.827   | 0.815    |
| 8x8 patches    | 1.0    | 0.310   | 0.324    | 0.3    | 0.711   | 0.705    |
| 4x4 patches    | 1.0    | 0.276   | 0.276    | 0.3*   | 0.606   | 0.602    |
*CNN lr for patch16/patch4 taken from the patch8 val result (time), MLP always full grid.
Note: MLP on patch8/patch4 picked lr 1.0, the top of the grid. Its val is low everywhere (0.24-0.31 over the
grid) and this condition is about the CNN, so the grid was not widened further.

### (a)
CNN 0.897 vs MLP 0.573 on test under the pre-registered criterion (+32.4 points). Same parameters, epochs and
tuning, so the gap comes from the architecture. Proposed reason (stated before results): use of the local spatial
arrangement of pixels.

### (b) Prediction check
- MLP: 0.573 -> 0.576 (+0.3). As predicted, no change beyond noise.
- CNN: 0.897 -> 0.561 (-33.6), slightly below the permuted MLP (0.576). As predicted, the CNN loses essentially
  its whole advantage when neighbourhoods are destroyed, although every pixel value is still there.
- This supports the reason from (a). It also shows the CNN has no advantage left once the arrangement is gone:
  its inductive bias then works against it a little (local filters on non-neighbours).

### (c) Local evidence (A) vs global layout (B)
- CNN accuracy falls gradually with patch size: 0.897 (intact) -> 0.815 (16) -> 0.705 (8) -> 0.602 (4) -> 0.561
  (single pixels = permutation). With 8x8 patches in a random order per image the CNN keeps 79% of its clean
  accuracy, and it is 38 points above the MLP under the same shuffle. Most of the CNN's accuracy comes from local
  evidence (A).
- Layout still matters: shuffling 8x8 patches costs 19 points. The MLP, in contrast, depends on fixed positions
  (0.573 -> 0.324) because its weights are tied to absolute pixel locations.
- Errors (8x8 vs clean CNN, per-class recall): biggest losses for cat -27, deer -26, bird -22, horse -22, dog -21;
  smallest for ship -10, truck -14, frog -16, automobile -16. Confusions that grow most: dog->cat 89->178,
  cat->dog 76->175, automobile->truck 34->108, truck->automobile 28->83, deer->frog 15->93, bird->frog 29->80.
  Pairs that share texture/colour and differ mainly in shape/configuration (cat/dog, car/truck) are the ones that
  need the layout; classes with distinctive local context (ship: water, frog: texture) survive.
- Confidence (test): the clean CNN is well calibrated (ECE 0.033; mean confidence 0.956 right / 0.710 wrong;
  25% of its errors have confidence > 0.9). Trained on 8x8 shuffles: ECE 0.164, confidence on errors goes UP to
  0.753 and 34% of errors are > 0.9 confident. So without the layout the CNN does not become uncertain, it stays
  confident from local evidence and is wrong more often. That is what a bag-of-local-features decision would do.
- Extra (not a train=test condition): the clean-trained CNN on 8x8-shuffled test images gets 0.277 with mean
  confidence 0.727 (ECE 0.45): when trained on intact images it does use the layout it was given and is
  confidently wrong without it. On 16x16 shuffles it gets 0.582.
- Conclusion for (c): the CNN's behaviour is mostly explained by local evidence (A), with global layout (B)
  needed to separate classes that look alike locally.

### Linking the three
(a) the CNN wins by 32 points at equal budget; (b) removing neighbourhoods removes the whole gain, so the gain
comes from spatial arrangement; (c) of that arrangement, local neighbourhoods (patches) carry most of it and
global layout a smaller part concentrated in shape-defined pairs.
