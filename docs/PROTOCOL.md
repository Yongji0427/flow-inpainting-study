# Experimental protocol

## Question and controls

The study compares pixel regression, perceptual regression, and conditional flow matching, then varies the NFE budget inside frozen FM models. All restoration CNNs use the same 14,986,435-parameter architecture, 30,000 updates, seed 0/1, and matched initialization and training data streams. Perceptual regression additionally uses a frozen VGG16 network during training.

## Data

The inherited image-level manifest contains 2,924 training, 730 validation, and 3,669 official-test images. It follows an earlier near-duplicate review of the train/validation pool. A fixed subset of 128 validation images supplies the development pool. After paired-mask support checks, 2,812 training, 124 development, and 3,585 test images remain. Neither duplicate checks nor the split definition guarantee animal-identity separation. These image sets were also used in an earlier classification project; the restoration CNNs are trained from scratch.

Images use EXIF correction, RGB conversion, bicubic center-fit resizing to 64 × 64, and normalization to `[-1,1]`. Training jointly flips images and masks horizontally with probability 0.5.

Each pair contains one large and one fragmented mask with exactly 1,638 missing pixels. Missing area, quadrant counts, and the central 16 × 16 mask pattern match exactly. Foreground and available head-box occlusion are matched within tolerances. Test matching uses trimaps and spatial constraints because local test head boxes are unavailable. Mask selection uses annotations, not model outputs.

## Selection and evaluation

| Choice | Selection or control |
|---|---|
| CNN width | Seed-0 development comparison of 1.75M, 6.73M and 14.99M FM models |
| FM sampling | NFE in {1,2,4,8,16,32}; select 8 by development LPIPS |
| Perceptual coefficient | Compare alpha in {0.1,1.0} on seed-0 development LPIPS; repeat only alpha=1 on seed 1 |
| Checkpoint | Predetermined final 30k EMA endpoint; intermediate checkpoints are diagnostic |
| Test | Fixed settings, four mask pairs per image, and four fixed FM noise samples per condition |

The mean development PSNR peaks at NFE2 and LPIPS is lowest at NFE8 among the tested budgets. The test set evaluates the fixed FM8 configuration, not a new NFE sweep. Perceptual regression is a subsequent control designed after the original FM/pixel-regression test results were available. Its own coefficient and checkpoints were fixed before its test evaluation.

The VGG feature loss averages L1 feature differences at feature indices 3, 8 and 15, with the initial magnitude calibration `c=0.33623046`. The selected training objective is pixel MSE plus `alpha*c` times feature loss. Evaluation LPIPS uses a different backbone, SqueezeNet.

## Training settings

AdamW uses batch size 64, weight decay 1e-4, default moment coefficients, gradient-norm clipping at 1, and EMA decay 0.995. The inherited learning-rate schedule is implemented in `GeoStep/train_matched.py::lr_at`. Development objective probes and task metrics are different measurements and may favor different checkpoints.

## Scope of conclusions

This is a controlled study on one low-resolution dataset, two mask geometries, and two trained seeds. Overall improvements do not imply a win on every image. The image-bootstrap intervals describe image-sampling uncertainty conditional on these models, not uncertainty over a population of independently trained models. Reference PSNR/LPIPS do not establish distributional realism or semantic diversity. Results do not establish that increasing integration steps increases numerical integration error.
