## Pretraining

## Introduction

This page outlines the steps to pretrain a model with T5X on common tasks
defined with [SeqIO](https://github.com/google/seqio/blob/main/README.md).

## Overview

Pretraining a model with T5X consists of the following steps:

1.  Choose the model architecture.
2.  Choose the SeqIO Task/Mixture to for training.
3.  Write a Gin file that configures the model, SeqIO Task/Mixture and other
    details of your pretraining run.
4.  Launch your experiment locally or on XManager.
5.  Monitor your experiment.

These steps are explained in detail in the following sections. An example run
that trains a T5 1.1 Small checkpoint from scratch on the C4 dataset using the
span corruption pretraining objective is also showcased.

## Step 1: Choose a model architecture

To train a model, you need a Gin config file that defines the model params. For
your convenience, Gin configs for common models have been made available for use
in T5X. Following is a list of these models and their Gin locations.

For the example run, you will use the T5 1.1 Small model. The Gin file for this
model is located at
[`/t5x/examples/t5/t5_1_1/1_1_small.gin`](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/small.gin).

## Step 2: Choose a SeqIO Task/Mixture

A SeqIO Task encapsulates the data source, the preprocessing logic to be
performed on the data before querying the model, the postprocessing logic to be
performed on model outputs, and the metrics to be computed given the
postprocessed outputs and targets. A SeqIO Mixture denotes a collection of Tasks
and enables pretraining a model on multiple Tasks simultaneously.

Many common datasets and benchmarks, e.g. [GLUE](https://gluebenchmark.com/),
[SuperGLUE](https://super.gluebenchmark.com/),
[WMT](https://www.tensorflow.org/datasets/catalog/wmt_t2t_translate),
[SQUAD](https://rajpurkar.github.io/SQuAD-explorer/),
[CNN/Daily Mail](https://github.com/abisee/cnn-dailymail), etc. have been
implemented as SeqIO Tasks/Mixtures and can be used directly. These
Tasks/Mixtures are defined in
[`third_party/py/t5/data/tasks.py`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/tasks.py)
and
[`third_party/py/t5/data/mixtures.py`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/mixtures.py).

For the example run, you will train the model on
[`c4_v220_span_corruption`](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/t5/data/tasks.py?l=42&rcl=370153959)
Task that implements the span corruption pretraining objective using the C4
dataset. This is the final pretraining Task used in the
[T5 paper](https://arxiv.org/pdf/1910.10683.pdf%C3%82%C2%A0).

TIP: Want to use a custom Task or Mixture? See section below called "Adding
SeqIO Task/Mixture modules and Gin files"

## Step 3: Write a Gin Config

After choosing the model architecture and SeqIO Task/Mixture for your run, the
next step is to configure your run using Gin. If you're not familiar with Gin,
reading the [T5X Gin Primer](gin.md) is recommended.

T5X provides a Gin file that configures the T5X trainer for pretraining (located
at
[`runs/pretrain.gin`](https://github.com/google-research/t5x/blob/main/t5x/configs/runs/pretrain.gin)),
and expects a few params from you. These params can be specified in a separate
Gin file, or via commandline flags. Following are the required params:

+   `TRAIN_STEPS`: Number of training steps. For the example run, set this to
    `100_000`.
+   `MIXTURE_OR_TASK_NAME`: This is the SeqIO Task or Mixture name to run (from
    Step 2). For the example run, set this to `'c4_v220_span_corruption'`.
+   `TASK_FEATURE_LENGTHS`: This is a dict mapping feature key to maximum int
    length for that feature. After preprocessing, features are truncated to the
    provided value. For the example run, set this to `{"inputs": 512, "targets":
    114}`, following the original T5 pretraining setup.
+   `MODEL_DIR`: A path to write pretrained checkpoints to. When launching using
    XManager, this path is automatically set and can be accessed from the
    XManager Artifacts page. When running locally using Blaze, you can
    explicitly pass a directory using a flag. Launch commands are provided in
    the next step.

In addition to the above params, you will need to import
[`pretrain.gin`](https://github.com/google-research/t5x/blob/main/t5x/configs/runs/pretrain.gin)
and the Gin file for the pretrained model, which for the example run is
[`t5_1_1/small.gin`](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/small.gin).

```gin
include 't5x/configs/runs/pretrain.gin'
include 't5x/examples/t5/t5_1_1/small.gin'
```

Note that the `include` statements can use relative paths in this example for
which You will pass an appropriate `gin_search_paths` flag to locate these files
when launching your run. However, we recommend that you use absolute paths
because it can be more difficult to locate the gin files speicified via relative
paths without inspecting the launch command.

You will also need to import the Python module(s) that register SeqIO Tasks and
Mixtures used in your run. For the example run, we add `import t5.data.mixtures`
since it is where 'glue_v002_proportional' is registered. Note that this module
must also be included as a dependency in the T5X trainer
[binary](https://github.com/google-research/t5x/blob/main/t5x/BUILD;l=74;rcl=398627055). Most
common Task/Mixture modules, such as this one, are already included.

Finally, your Gin file should look like this:

```gin
include 't5x/examples/t5/t5_1_1/small.gin'
include 't5x/configs/runs/pretrain.gin'

# Register necessary SeqIO Tasks/Mixtures.
import t5.data.mixtures

MIXTURE_OR_TASK_NAME = "c4_v220_span_corruption"
TASK_FEATURE_LENGTHS = {"inputs": 512, "targets": 114}
TRAIN_STEPS = 10000
DROPOUT_RATE = 0.0
BATCH_SIZE = 256
```

See
[`t5x/examples/t5/t5_1_1/examples/small_c4_pretrain.gin`](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/examples/small_c4_pretrain.gin)
for this example.


## Step 4: Launch your experiment

To launch your experiment locally (for debugging only; larger checkpoints may
cause issues), run the following on commandline:

```sh
MODEL_DIR="/tmp/pretrain-model/"
python -m t5x.train \
  --gin_file=t5x/examples/t5/t5_1_1/c4_pretrain_small.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr
```

Note that multiple comma-separated paths can be passed to the `gin_search_paths`
flag, and these paths should contain all Gin files used or included in your
experiment.