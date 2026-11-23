# Q3 notes

## (a) Three convolutional autoencoders
- Architecture (Q3/models.py): 3 stride-2 conv layers (32, 64, 128 channels, 4x4 kernels, BN, ReLU) down to
  128x4x4, a linear layer to the bottleneck, and the mirror image with transposed convs. Only the two
  bottleneck linear layers change between models, so the conv part is identical.
- Bottlenecks 16, 128, 1024 (factor 8 apart; 192x, 24x and 3x compression of the 3072 input values).
  Parameter counts: 398,995 / 857,859 / 4,528,771. The difference is entirely in the two linear layers.
- Same training for all: AdamW lr 1e-3, wd 1e-5, OneCycle, 30 epochs, batch 256, MSE in the normalised
  input space, seed 242003, no labels. Train MSE and val MSE logged each epoch (curves are flat by epoch 30).
- Working in the normalised space means reconstructions can be passed directly to the Q1 CNN. MSE is
  also reported in [0,1] pixel units (and PSNR) because that is easier to interpret.
- Selection by val reconstruction error only: d=1024 has the lowest val MSE (pixel 0.00194 vs 0.00502 and
  0.01650). If reconstruction is the only criterion there is nothing to trade off: the error keeps falling as
  the bottleneck grows, so the largest bottleneck is the choice. It is still far from an identity map (PSNR
  about 27 dB, fine texture is blurred).

## (b) Denoising
- Corruption: additive Gaussian noise, std 0.1 in [0,1] pixel units, clipped to [0,1]. Reconstruction can
  reasonably help because the noise is independent per pixel while images are smooth and correlated, so a
  model of natural images can separate the two. A blur or occlusion would remove information that a
  reconstruction can only guess.
- DAE: the same architecture and bottleneck as the selected clean AE (1024), trained with noisy input and the
  clean image as target, noise redrawn every batch, same optimiser/schedule/epochs. No labels.
  So the clean AE vs DAE comparison differs only in the training input.
- Fixed classifier: the clean CNN from Q1(a) (reused, lr chosen there on val). It is never retrained.
- Test noise: drawn once with a fixed seed. Also evaluated at std 0.05 and 0.2 to see whether the DAE only
  works at the noise level it was trained on.
- Controls: the classifier on clean-AE(clean) and DAE(clean) shows how much the reconstruction step itself
  costs even with no noise.

## (c) Class information vs reconstruction
- Frozen encoders, codes taken from the bottleneck. Two probes so the conclusion does not depend on one:
  a linear probe (multinomial logistic regression, LBFGS, features standardised with train statistics,
  L2 weight from {1e-5,...,1e-1} chosen on val) and k-NN with cosine similarity (k from {1,...,100} chosen on
  val). The first grids were {1e-5..1e-2} and k up to 50; raw pixels picked 1e-2 and AE-16 picked k=50,
  both at the edge, so the grids were widened. The selected values did not change.
- Raw pixels are included as a reference.

## Results (Q3/results/q3_summary.json)

### (a)
| bottleneck | params    | val MSE (norm.) | val MSE (pixel) |
|------------|-----------|-----------------|-----------------|
| 16         | 398,995   | 0.2633          | 0.01650         |
| 128        | 857,859   | 0.0800          | 0.00502         |
| 1024       | 4,528,771 | 0.0308          | 0.00194         |
Selected: 1024. Test MSE (pixel) 0.00194, PSNR 27.1 dB. Figure: figures/q3_reconstructions.

### (b) Fixed classifier = Q1(a) CNN (lr 0.3, clean test accuracy 0.897)
Test accuracy of the classifier, and in brackets the MSE ([0,1] pixels) of its input to the clean image:

| input                | std 0.05        | std 0.1 (trained) | std 0.2         |
|----------------------|-----------------|-------------------|-----------------|
| noisy image          | 0.655 (0.00242) | 0.280 (0.00931)   | 0.115 (0.0333)  |
| clean AE(noisy)      | 0.616 (0.00226) | 0.553 (0.00319)   | 0.406 (0.00670) |
| DAE(noisy)           | 0.548 (0.00254) | 0.524 (0.00296)   | 0.422 (0.00512) |
| control: clean AE(clean) | 0.646 (0.00194) |               |                 |
| control: DAE(clean)  | 0.553 (0.00243) |                   |                 |

Interpretation:
- At the noise level it was trained for (std 0.1), denoising brings the classifier from 0.280 back to 0.524,
  so it recovers a large part of what the noise destroyed, but far from the clean 0.897.
- The control rows show why: passing a CLEAN image through either autoencoder already drops accuracy to
  0.646 (AE) / 0.553 (DAE). The autoencoders keep the coarse layout and colours but smooth out the fine texture
  and edges that the CNN relies on. So the reconstructions do not contain all the information the classifier
  needs, noise or no noise. With these autoencoders about 0.65 is the ceiling for any reconstruct-then-classify
  approach.
- The DAE produces outputs closer to the clean image in MSE (0.00296 vs 0.00319 at std 0.1) but the classifier
  does worse on them (0.524 vs 0.553). Lower pixel error does not mean more of the information the classifier
  uses: the DAE learned to smooth more aggressively (good for MSE against noise, bad for texture).
- The DAE only wins at a stronger noise (std 0.2: 0.422 vs 0.406), where the clean AE passes more noise through.
  At weak noise (0.05) feeding the noisy image directly (0.655) is better than either reconstruction.
- Conclusion: denoising recovers part of the class information (+24 points at std 0.1), but not the information
  the classifier needs to get near its clean accuracy, because the reconstruction step itself removes it.
  The classifier was trained on sharp clean images and is fixed, so part of the loss is a distribution shift
  (blurry inputs); retraining the classifier on reconstructions would be a different experiment and is not
  allowed by the question.

### (c) Frozen codes (test set)
| encoder  | dim  | test MSE (pixel) | PSNR | linear probe (wd) | k-NN (k)   |
|----------|------|------------------|------|-------------------|------------|
| AE-16    | 16   | 0.01647          | 17.8 | 0.331 (1e-4)      | 0.422 (50) |
| AE-128   | 128  | 0.00499          | 23.0 | 0.410 (1e-3)      | 0.409 (10) |
| AE-1024  | 1024 | 0.00194          | 27.1 | 0.532 (1e-4)      | 0.485 (20) |
| DAE-1024 | 1024 | 0.00243          | 26.1 | 0.550 (1e-4)      | 0.490 (10) |
| raw pixels | 3072 | -              | -    | 0.415 (1e-2)      | 0.426 (10) |

Interpretation:
- Across the three clean AEs, better reconstruction goes with more linearly available class information
  (0.331 -> 0.410 -> 0.532), so at this coarse level the two agree.
- They disagree in two places. (1) The DAE reconstructs clean images worse than AE-1024 (0.00243 vs 0.00194)
  but its code holds more class information (0.550 vs 0.532 linear; the k-NN gap, 0.490 vs 0.485, is too
  small to count with one seed). A likely reason: to remove noise the code has to keep structure that is stable
  under small pixel changes, and that kind of structure is more class related than exact pixel values.
  (2) With k-NN, AE-16 (0.422) beats AE-128 (0.409) even though it reconstructs 3x worse. Our guess is that
  the 16-d code keeps mostly global colour/layout, which nearest neighbours can use, while the extra
  dimensions of AE-128 encode detail that adds noise to the distances. (Linear probe ranks them the other way.)
- The 128-d code is not better than raw pixels at all, so compression alone does not create class
  information. Only AE-1024 / DAE-1024 beat raw pixels, by 12-14 points linearly.
- Conclusion: reconstruction quality and class information agree only roughly; reconstruction error is not a
  reliable measure of how useful a representation is for classification.
