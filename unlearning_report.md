# Anti biomimetic training for unlearning

## Intro

This report follows the suggestion to use anti biomimetic training as an unlearning tool. I start from a biomimetic trained checkpoint, then continue training using a reversed blur curriculum. I track clean accuracy and corruption robustness across epochs.

## Motivation

Biomimetic blur schedules push the network toward lower frequency structure early in training. That can act like a bias toward global cues. The unlearning idea is that a reversed schedule can shift the learned representation and reduce the robustness behavior that biomimetic training encouraged. This is a first pass test of that sequential adaptation idea on CIFAR 10.

## Experimental

## Experimental question

Starting from a biomimetic checkpoint, how do clean CIFAR 10 accuracy and CIFAR 10 C accuracy change during fine tuning under an anti biomimetic increasing blur schedule.

## Set up

Model is the C KAN in this repository.

Start checkpoint is `data1/model_b_biomimetic.pth`.

Fine tuning runs for 20 epochs.

Training augmentation is Gaussian blur on the training images.

Sigma increases linearly from 0.0 to 3.8 in steps of 0.2.

Each epoch I evaluate clean CIFAR 10 test accuracy.

Each epoch I evaluate CIFAR 10 C mean accuracy over four corruptions at severity 3. The corruptions are snow, glass blur, defocus blur, fog.

Final checkpoint is `data1/model_unlearned_from_b.pth`.

The purpose of this setup is to isolate the effect of the reversed blur curriculum during sequential training. The architecture stays fixed and the starting weights stay fixed. The only planned change is the training data blur schedule during fine tuning. Measuring the same two metrics after every epoch gives a curve rather than a single number.

Clean CIFAR 10 accuracy is included because it is the simplest reference point for whether the model still solves the original task. CIFAR 10 C mean accuracy is included because it is a standard way to measure robustness under common corruptions.

## Numerical results

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

## Analysis

The baseline at epoch 0 has clean accuracy 64.43 percent and CIFAR 10 C mean 57.61 percent.

During the first few epochs, sigma stays small and the model still trains on mostly sharp images. In this regime the corruption mean improves slightly and reaches 58.82 percent at epoch 4. Clean accuracy stays close to the starting point.

After that point sigma continues to increase every epoch and the training distribution shifts toward heavier blur. Clean accuracy decreases steadily from the mid 60s down to the low 40s by the end. The corruption mean also decreases from the high 50s down to about 40.

At epoch 20, clean accuracy is 42.13 percent and the corruption mean is 40.41 percent.

One way to read this is that there are two phases. A short early phase where mild blur fine tuning does not hurt and can slightly help the chosen corruption subset. A longer phase where strong blur dominates training and performance drops on both clean data and corrupted data.

## Discussion

This run supports the claim that consistent tuning under an anti biomimetic region changes the model behavior over time. As blur strength increases, both clean accuracy and the evaluated robustness metric decrease.

For the unlearning framing, this suggests that the robustness learned under biomimetic is sensitive to later training distribution shifts. It also suggests that unlearning in this specific implementation is not selective, because the same intervention that reduces the corruption mean also reduces clean accuracy.

The small early improvement is interesting because it suggests that continued training from the biomimetic checkpoint can briefly improve the corruption subset score even when sigma remains near zero. That could be simple continued optimization, or it could be that mild blur acts like a regularizer that matches the blur related corruptions in the evaluation set.

## Scope and next steps

The robustness metric here is a mean over four corruptions at one severity. A stronger follow up uses more CIFAR 10 C corruptions and multiple severities.

Another follow up runs the same sequential procedure on a size matched CNN and compares the curves. If the curves differ, that supports an architecture dependent unlearning dynamic.

A final follow up defines a clear second objective to recover, then repeats the sequential procedure while tracking that objective. That makes the unlearning claim closer to the original idea of trading robustness for performance in a different setting.
