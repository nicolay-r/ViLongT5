# Fine Tuning a Model


## Introduction

This page outlines the steps to fine-tune an existing pre-trained model with T5X
on common downstream tasks defined with [SeqIO](https://github.com/google/seqio/blob/main/README.md). This is one of
the simplest and most common use cases of T5X. If you're new to T5X, this
tutorial is the recommended starting point.

## Overview

Fine-tuning a model with T5X consists of the following steps:

1.  Choose the pre-trained model to fine-tune.
2.  Choose the SeqIO Task/Mixture to fine-tune the model on.
3.  Write a Gin file that configures the pre-trained model, SeqIO Task/Mixture
    and other details of your fine-tuning run.
4.  Launch your experiment locally or on XManager.

These steps are explained in detail in the following sections. An example run
that fine-tunes a T5-small checkpoint on WMT14 English to German translation
benchmark is also showcased.

## Step 1: Choose a pre-trained model

To use a pre-trained model, you need a Gin config file that defines the model
params, and the model checkpoint to load from. For your convenience, TensorFlow
checkpoints and Gin configs for common T5 pre-trained models have been made
available for use in T5X. A list of all the available pre-trained models (with
model checkpoints and Gin config files) are available in the
[Models](https://github.com/google-research/t5x/blob/main/docs/models.md) documentation.

For the example run, you will use the T5 1.1 Small model. The Gin file for this
model is located at
[`/t5x/examples/t5/t5_1_1/small.gin`](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/small.gin),
and the checkpoint is located at
[`gs://t5-data/pretrained_models/t5x/t5_1_1_small`](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/t5_1_1_small).

## Step 2: Choose a SeqIO Task/Mixture

A SeqIO Task encapsulates the data source, the preprocessing logic to be
performed on the data before querying the model, the postprocessing logic to be
performed on model outputs, and the metrics to be computed given the
postprocessed outputs and targets. A SeqIO Mixture denotes a collection of Tasks
and enables fine-tuning a model on multiple Tasks simultaneously.

## Step 3: Write a Gin Config

After choosing the pre-trained model and SeqIO Task/Mixture for your run, the
next step is to configure your run using Gin. If you're not familiar with Gin,
reading the [T5X Gin Primer](gin.md) is recommended.

T5X provides a Gin file that configures the T5X trainer for fine-tuning (located
at
[`t5x/configs/runs/finetune.gin`](https://github.com/google-research/t5x/blob/main/t5x/configs/runs/finetune.gin)),
and expects a few params from you. These params can be specified in a separate
Gin file, or via commandline flags. Following are the required params:

+   `INITIAL_CHECKPOINT_PATH`: This is the path to the pre-trained checkpoint
    (from Step 1). For the example run, set this to
    `'gs://t5-data/pretrained_models/t5x/t5_1_1_small/checkpoint_1000000'`.
+   `TRAIN_STEPS`: Number of fine-tuning steps. This includes the number of
    steps that the model was pre-trained for, so make sure to add the step
    number from the `INITIAL_CHECKPOINT_PATH`. For the example run, to fine-tune
    for `20_000` steps, set this to `1_020_000`, since the initial checkpoint is
    the `1_000_000`th step.
+   `MIXTURE_OR_TASK_NAME`: This is the SeqIO Task or Mixture name to run (from
    Step 2). For the example run, set this to `'wmt_t2t_ende_v003'`.
+   `TASK_FEATURE_LENGTHS`: This is a dict mapping feature key to maximum int
    length for that feature. After preprocessing, features are truncated to the
    provided value. For the example run, set this to `{'inputs': 256, 'targets':
    256}`.
+   `MODEL_DIR`: A path to write fine-tuned checkpoints to. When launching using
    XManager, this path is automatically set and can be accessed from the
    XManager Artifacts page. When running locally using Blaze, you can
    explicitly pass a directory using a flag. Launch commands are provided in
    the next step.
+   `LOSS_NORMALIZING_FACTOR`: When fine-tuning a model that was pre-trained
    using Mesh Tensorflow (e.g. the public T5 / mT5 / ByT5 models), this should
    be set to `pretraining batch_size` * `pretrained target_token_length`. For
    T5 and T5.1.1: `2048 * 114`. For mT5: `1024 * 229`. For ByT5: `1024 * 189`.

In addition to the above params, you will need to include
[`finetune.gin`](https://github.com/google-research/t5x/blob/main/t5x/configs/runs/finetune.gin)
and the Gin file for the pre-trained model, which for the example run is
[`t5_1_1/small.gin`](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/small.gin).

```gin
include 't5x/configs/runs/finetune.gin'
include 't5x/examples/t5/t5_1_1/small.gin'
```

You will also need to import the Python module(s) that register SeqIO Tasks and
Mixtures used in your run. For the example run, we add `import t5.data.tasks`
since it is where `wmt_t2t_ende_v003` is registered.


Finally, your Gin file should look like this:

```gin
include 't5x/configs/runs/finetune.gin'
include 't5x/examples/t5/t5_1_1/small.gin'

# Register necessary SeqIO Tasks/Mixtures.
import t5.data.tasks

MIXTURE_OR_TASK_NAME = "wmt_t2t_ende_v003"
TASK_FEATURE_LENGTHS = {"inputs": 256, "targets": 256}
TRAIN_STEPS = 1_020_000  # 1000000 pre-trained steps + 20000 fine-tuning steps.
DROPOUT_RATE = 0.0
INITIAL_CHECKPOINT_PATH = "gs://t5-data/pretrained_models/t5x/t5_1_1_small/checkpoint_1000000"
LOSS_NORMALIZING_FACTOR = 233472
```

See
[`t5x/examples/t5/t5_1_1/examples/small_wmt_finetune.gin`](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/examples/small_wmt_finetune.gin)
for this example.


## Step 4: Launch your experiment

To launch your experiment locally (for debugging only; larger checkpoints may
cause issues), run the following on commandline:

```sh
MODEL_DIR="/tmp/finetune-model/"
python -m t5x.train \
  --gin_file=t5x/examples/t5/t5_1_1/examples/small_wmt_finetune.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr
```

Note that multiple comma-separated paths can be passed to the `gin_search_paths`
flag, and these paths should contain all Gin files used or included in your
experiment.


After fine-tuning has completed, you can parse metrics into CSV format using the
following script:

```sh
MODEL_DIR= # from Step 4 if running locally, from XManager Artifacts otherwise
VAL_DIR="$MODEL_DIR/inference_eval"
python -m t5.scripts.parse_tb \
  --summary_dir="$VAL_DIR" \
  --seqio_summaries \
  --out_file="$VAL_DIR/results.csv" \
  --alsologtostderr
```
