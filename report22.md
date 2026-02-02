Hey!

Following clarification that you meant anti‑biomimetic as an unlearning tool (sequential adaptation), I ran the experiment in that form and wrote up a  report.

## 1) Goal / question tested

- Starting point-;W a CKAN .pth file already done w/ past training
- Intervention (“unlearning”): continue training from those weights, but switch the data to **anti‑biomimetic** (increasing blur), to test whether robustness gains can be “forgotten” (reduced) and whether another metric can be recovered/improved.
- Metrics tracked per epoch:
  - CIFAR‑10 test accuracy 
  - CIFAR‑10‑C robustness mean across a fixed corruption subset

## 2) Code + reproducibility pointers

- unlearning script: `scripts/unlearn_biomimetic.py`
- report output `unlearning_report.md`
- Starting checkpoint used `data1/model_b_biomimetic.pth`
- Final checkpoint saved `data1/model_unlearned_from_b.pth`
- Metrics JSON saved during run  `data1/unlearning_metrics.json` (written by script)

## 3) Model + data details

- Architecture: Convolutional KAN implemented with KAN convolution layers 
- Dataset (clean): CIFAR‑10 test set, standard normalization used by the repo:
  - mean = (0.4914, 0.4822, 0.4465)
  - std  = (0.2023, 0.1994, 0.2010)
  - Severity used 3
  - Corruptions evaluated (fixed set of 4 for now): `snow`, `glass_blur`, `defocus_blur`, `fog`

## 4) “Unlearning” intervention (anti‑biomimetic fine‑tuning schedule)

- Total fine‑tune epochs: 20
- Blur augmentation during training: Gaussian blur applied to training images with sigma given by a schedule.
- Sigma schedule used in this run (as reflected by the report table): linear ramp from 0 to 3.8 over epochs 0..20, with step 0.2. Concretely, the table’s sigma values were:
  - epoch 0–1: sigma = 0.0
  - epoch 2: 0.2
  - …
  - epoch 20: 3.8
- GaussianBlur kernel sizing: `kernel_size = ceil(4*sigma)`, forced to be odd (`kernel_size += 1` if even).
## 6) Results (complete table)

The numbers below are exactly what the run produced and what is recorded in the report.

| epoch | sigma | clean_acc (%) | c10c_mean_4 (%) |
|------:|------:|--------------:|----------------:|
| 0 | 0.000 | 64.43 | 57.61 |
| 1 | 0.000 | 65.78 | 58.00 |
| 2 | 0.200 | 65.69 | 58.13 |
| 3 | 0.400 | 65.53 | 58.37 |
| 4 | 0.600 | 64.00 | 58.82 |
| 5 | 0.800 | 60.37 | 56.80 |
| 6 | 1.000 | 58.01 | 55.57 |
| 7 | 1.200 | 55.46 | 53.40 |
| 8 | 1.400 | 52.26 | 50.73 |
| 9 | 1.600 | 51.79 | 50.66 |
| 10 | 1.800 | 49.01 | 48.51 |
| 11 | 2.000 | 48.74 | 47.70 |
| 12 | 2.200 | 47.41 | 46.35 |
| 13 | 2.400 | 45.88 | 45.16 |
| 14 | 2.600 | 45.30 | 44.52 |
| 15 | 2.800 | 44.27 | 43.51 |
| 16 | 3.000 | 44.07 | 42.98 |
| 17 | 3.200 | 43.56 | 42.24 |
| 18 | 3.400 | 43.29 | 42.06 |
| 19 | 3.600 | 41.92 | 40.27 |
| 20 | 3.800 | 42.13 | 40.41 |

## 7) Key observations (directly from the table)

- **Baseline (epoch 0; biomimetic checkpoint before unlearning):**
  - clean_acc = 64.43%
  - c10c_mean_4 = 57.61%
- Very early “unlearning” (sigma still ~0 to 0.6): robustness does *not* immediately drop; it slightly increases up to 58.82% at epoch 4 while clean accuracy stays ~64–66%.
- Once sigma becomes moderate (≈0.8 and above), both metrics trend downward together.
- Final state (epoch 20; sigma=3.8):
  - clean_acc = 42.13% (−22.30 points from baseline)
  - c10c_mean_4 = 40.41% (−17.20 points from baseline)

## 8) Interpretation (what this does and does not show)

- This sequential adaptation setup *does* demonstrate that pushing the training distribution toward heavy blur can substantially degrade the model’s clean accuracy and also reduce themean.
- In this run, robustness was not “selectively forgotten” in a way that recovered clean accuracy; instead, after a certain blur level, I can observe broad degradation (clean and robustness both falling).
- The early‑epoch bump in robustness (epochs 1–4) suggests there is a short regime where continued training does not immediately erase robustness, but the longer anti‑biomimetic ramp leads to significant performance collapse on both metrics.
