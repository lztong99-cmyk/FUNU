# README

# Necessary Unlearning Experiment README

This document describes the execution workflow and key details of the FUNU project for MNIST, CIFAR-10, and CIFAR-100 datasets, based on the `run.sh` script.

## Overview

The experiments focus on filter-based unlearning data selection across different datasets (MNIST, CIFAR-10, CIFAR-100) with varying hyperparameters (e.g., unlearning ratios, class-based unlearning, sample size thresholds). The workflow uses a Conda environment and executes Python scripts with YAML configuration files to run different experimental setups.

## Prerequisites

### 1. Conda Environment

Ensure the specified Conda environment is created and activated:

```Bash

conda activate your_environment_name
pip install -r requirements.txt
```

This environment is configured with Python 3.8, PyTorch 1.8.1, and CUDA 11.1.

### 2. File Structure

The project root is `~/yourfolder`, with the following key structure:

```Plain Text

yourfolder/
├── config/                # YAML configuration files for experiments
├── src/                   # Source code (main.py)
└── run.sh                 # Execution script (this file)
```

## Execution

All experiments are executed from the `src/` directory. Users can simply running the following command:

```
nohup python main.py --config ../config/filter_rfmodel_CIFAR100_100.yml > nohup.out
```

## Configuration Files

Each YAML config file (e.g., `filter_rfmodel_MNIST_0.1.yml`) defines dataset-specific parameters, unlearning ratios, model hyperparameters, and output paths. Modify these files to adjust experimental settings.

|Parameter|Value Example|Explanation|
|---|---|---|
|`dataset_path`|`"../dataset"`|Path to the root directory where the dataset is stored (relative to the script using this YAML).|
|`model_save_path`|`"../models"`|Directory to save trained/filtered/unlearned models (checkpoints, weights).|
|`log_path`|`"../log/"`|Path to store experiment logs (e.g., training metrics, filter results).|
|`attack_path`|`"../MIA/attack"`|Path to files related to Membership Inference Attack (MIA).|
|`device`|`"cuda:0"`|Computation device.|
|`dataset`|`"CIFAR10"`|Target dataset for the experiment (options: MNIST, CIFAR10, CIFAR100).|
|`model`|`"ResNet-18"`|Neural network architecture (options: 2-layer-CNN, ResNet-34). ResNet-18.|
|`model_pretrained`|`true`|Whether to use a pretrained model.|
|`original_training_exp`|`false`|Whether to run a baseline training experiment (train the model from scratch). `false` means the script skips original training.|
|`training_test_split`|`0.1`|Fraction of the dataset to use as a test set.|
|`model_original_path`|`false`|Path to load a pre-trained "original model" (before filtering/unlearning). `false` means no preloaded model (or the parameter is disabled).|
|`learning_rate`|`0.0001`|Optimizer learning rate.|
|`batch_size`|`128`|Number of samples per batch during training/evaluation.|
|`epochs`|`60`|Total number of training epochs (full passes over the dataset) for the model.|
|`unlearning_request_filter_exp`|`true`|Whether to enable the filter-based unlearning request experiment (core flag for this section — `true` means the script runs filter-related logic).|
|`filter_method`|`["rfmodel", "clustering", "confidence", "curvature"]`|List of filtering methods to test: <br> - `rfmodel`: FUNU method. <br> - `clustering`: Clustering-based method. <br> - `confidence`: Confidence-based method. <br> - `curvature`: Curvature-based method.|
|`eval_filtered_model`|`true`|Whether to evaluate the performance of the unlearned/retrained model (e.g., accuracy, privacy metrics like MIA resistance).|
|`score_thres_dict`|`{"clustering": -3, "confidence": -3, "curvature": -3}`|Threshold scores for each filter method: samples with scores below/above this threshold are selected for unlearning. Negative values suggest the default score scale 
|`retraining_exp`|`false`|Whether to run a retraining experiment after filtering (`false` means retraining is disabled — focus is on filtering alone).|
|`unlearning_data_selection`|`"Random"`|Method to select data for unlearning (options: Random, clustering, confidence, curvature, representation — commented). `"Random"` is the default (baseline for comparing filter methods).|
|`unlearning_proportion`|`100`|Float (e.g., 0.1): Percentage/fraction of data to unlearn. Integer (e.g., 100): numbers of data to unlearn.|
|`unlearning_idx`|`-1`|Index of the unlearning subset ( `-1` typically means "no specific index").|
