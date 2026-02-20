# Report

# CIFAR 10 experiments on standard training biomimetic training anti biomimetic training and unlearning

## Purpose and motivation

I wrote this detailed report to summarize the set of experiments you requested regarding the properties of biomimetic training curriculums compared to standard and anti biomimetic approaches. The central motivation for this work was your question about whether anti biomimetic training could serve as an unlearning tool. Specifically, you asked if we could start with a robust biomimetic model and then use an anti biomimetic curriculum to forget the robustness learned for task A and potentially regain performance for a different task B.

In this document, I provide a comprehensive breakdown of the methodology, the specific numerical results from my executions, and my analysis of what these measurements imply about the plasticity of the Convolutional KAN or CKAN architecture under different data curriculums. My goal is to allow you to see the exact progression of the model behavior so we can decide on the next phase of the project.

## Data sources

I have synthesized the data for this report from two primary logs generated during the execution of the code. The first source is `results.txt`, which contains the printed console output for the robustness evaluation tables and the epoch by epoch progress logs during the unlearning phase. The second source is `data1/unlearning_metrics.json`, which is the machine readable record of the sequential fine tuning experiment and contains the precise granular data for the per corruption breakdown at every epoch.

## Common evaluation slice for robustness

To ensure that all comparisons between the different training regimens are valid and fair, I established a fixed evaluation protocol for robustness. For every experiment reported below, I evaluated the model on the CIFAR 10 C dataset at severity level 3. I focused on a specific subset of four corruptions. These corruptions are snow, glass blur, defocus blur, and fog. I selected these because they represent a mix of blur related and weather related artifacts that are relevant to the biomimetic hypothesis, which concerns how models process spatial frequency information.

## Experiment 1 comparisons of standard biomimetic and anti biomimetic training from scratch

### Models and checkpoints

I began by training three separate instances of the Convolutional KAN model to serve as our baselines. I want to be explicit about the architecture used here. I used the Convolutional KAN implementation from this repository, which is not a standard CNN. Instead, it constructs convolution operations using the Kolmogorov Arnold representation theorem. Specifically, the convolution layers are implemented using an Unfold operation to extract patches, followed by `KANLinear` operations on those patches. These `KANLinear` layers use B spline basis functions and learnable coefficients rather than standard scalar weights. This architecture is statistically distinct from standard kernel attention networks or multilayer perceptrons.

I trained each model from random initialization to ensure we could observe the pure effect of the training curriculum.

The first model is the standard model, saved as `data1/model_a_standard.pth`. This model was trained on the standard CIFAR 10 dataset with conventional augmentation and serves as the control group.

The second model is the biomimetic model, saved as `data1/model_b_biomimetic.pth`. This model was trained using a decreasing blur schedule, where images start heavily blurred and gradually become sharper over the course of training. This mimics the development of visual acuity in biological systems.

The third model is the anti biomimetic model, saved as `data1/model_c_antibiomimetic.pth`. This model was trained using an increasing blur schedule, where images start sharp and gradually become more blurred. This is the reverse of the biological curriculum.

### Numerical results on CIFAR 10 C severity 3

I evaluated all three models on the fixed robustness slice. The values below are taken directly from the robustness exam results logged in the results file.

| corruption | standard | biomimetic | anti biomimetic |
|---|---:|---:|---:|
| snow | 54.8% | 58.6% | 28.0% |
| glass blur | 57.5% | 60.9% | 31.2% |
| defocus blur | 59.2% | 61.8% | 35.3% |
| fog | 49.2% | 49.1% | 33.0% |
| mean over 4 corruptions | 55.19% | 57.61% | 31.88% |

### Comment and interpretation

The results from this first experiment demonstrate a clear hierarchy in performance on this evaluation slice. The biomimetic model is the strongest overall performer with a mean robustness score of 57.61 percent. It outperforms the standard model on three out of the four measured corruptions, specifically snow, glass blur, and defocus blur. This supports the hypothesis that training with a coarse to fine curriculum induces representations that are more resilient to spatial degradations.

The standard model performs respectably with a mean of 55.19 percent, but it consistently lags behind the biomimetic model in the blur related categories. The anti biomimetic model trained from scratch performs substantially worse than the other two, with a mean score of only 31.88 percent. This suggests that the increasing blur curriculum, when applied from initialization, prevents the model from learning a robust solution for these corruptions.

## Experiment 2 unlearning by sequential fine tuning from the biomimetic checkpoint under increasing blur

### Experimental detailed setup

After establishing the baselines, I implemented the sequential adaptation experiment you described. The experimental question was whether we could take the robust biomimetic model and fine tune it using an anti biomimetic schedule to change its behavior. In this context, task A is the robustness capability measured on CIFAR 10 C, and task B is the capability to classify clean CIFAR 10 images.

I started this run by loading the weights from `data1/model_b_biomimetic.pth`. I then continued training for a total of 20 epochs. During this fine tuning phase, I applied a Gaussian blur augmentation to the training images. I followed an increasing sigma schedule where the sigma value started at 0.0 and increased linearly in steps of 0.2 until it reached a maximum of 3.8 at the final epoch.

To track the unlearning process, I performed a full evaluation at the end of every single epoch. I measured the clean CIFAR 10 test accuracy to see if the model retained its primary classification ability. Simultaneously, I measured the CIFAR 10 C mean accuracy over the same four corruption slice at severity 3 to see if the robustness was continuously being unlearned. I saved the final resulting model as `data1/model_unlearned_from_b.pth`.

### Numerical results table

The following table presents the complete epoch by epoch evolution of the model metrics. Epoch 0 represents the state of the model before any fine tuning updates were applied.

| epoch | sigma | clean acc (%) | c10c mean 4 (%) |
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

## Analysis of the unlearning dynamics

I observed distinct phases in the model behavior during the fine tuning process. At epoch 0, before the fine tuning began, the clean accuracy was 64.43 percent and the CIFAR 10 C mean was 57.61 percent. This confirms that we started with a healthy and robust model.

During the early phase, covering epochs 1 through 4, the sigma value was relatively low, ranging from 0.0 to 0.6. In this regime, the model behavior was stable or even slightly improving. The clean accuracy remained in the mid sixties. Interestingly, the CIFAR 10 C mean increased slightly, reaching a peak of 58.82 percent at epoch 4. This suggests that a mild amount of blur training on top of the biomimetic weights does not immediately degrade robustness and might even act as a beneficial regularizer for the specific corruptions we measured.

During the later phase, spanning epochs 5 through 20, the sigma value increased from 0.8 up to 3.8. In this regime, I observed a steady and concurrent degradation of both metrics. The clean accuracy dropped significantly from the sixties down to 42.13 percent. The CIFAR 10 C mean followed a similar trajectory, falling from the upper fifties down to 40.41 percent.

### Endpoint breakdown at epoch 20

To understand the final state of the model, I extracted the per corruption accuracy breakdown at epoch 20 from the metrics file.

| corruption | accuracy at epoch 20 |
|---|---:|
| snow | 33.78% |
| glass blur | 41.27% |
| defocus blur | 45.29% |
| fog | 41.28% |
| mean over 4 corruptions | 40.41% |

Comparing this to the baseline, the performance on snow dropped from 58.6 percent to 33.78 percent. Glass blur dropped from 60.9 percent to 41.27 percent. Defocus blur dropped from 61.8 percent to 45.29 percent. Fog dropped from 49.1 percent to 41.28 percent.

## Conclusions supported by these measurements

Based on the data gathered from these experiments, I can draw several conclusions to answer your initial questions. First, regarding the baseline capability, the biomimetic training procedure produced the strongest robustness mean among the three independently trained models on the fixed CIFAR 10 C slice I evaluated. This confirms that the biomimetic curriculum provides a measurable robustness benefit compared to standard training.

Second, regarding the unlearning hypothesis, the experiment demonstrates that fine tuning the biomimetic model under an increasing blur schedule does indeed change the model behavior. The robustness mean eventually decreased substantially. However, the clean accuracy also decreased substantially.

In this specific implementation of the unlearning procedure, I did not observe a selective tradeoff where the model forgot its robustness while maintaining or recovering its clean accuracy. Instead, both capabilities degraded together as the blur strength increased. This suggests that the anti biomimetic curriculum, when applied with this intensity, acts destructively on the representations learned by the biomimetic model rather than selectively editing them.

## Limitations and future work

It is important to note the limitations of these results. These conclusions are based on a single severity level and a subset of four corruptions. The results might differ if we evaluated on a broader range of corruptions or different severity levels. Additionally, while we can infer behavioral changes from the accuracy metrics, we did not directly measure internal representation changes or frequency bias in these runs.

For the next steps, I propose we could explore modifications to the unlearning schedule to see if we can achieve a more selective effect. We strictly tied the sigma increase to the epoch count in this run. We could try a slower ramp or a lower maximum sigma to find a sweet spot where robustness is reduced without collapsing the clean performance. I can also run the same sequential procedure on a size matched CNN to determine if this degradation profile is specific to the Convolutional KAN architecture.

