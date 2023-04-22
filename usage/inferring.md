# Running inference on a Model

This page outlines the steps to run inference a model with T5X on Tasks/Mixtures
defined with [SeqIO](https://github.com/google/seqio/blob/main/README.md).

## Overview

Running inference on a model with T5X using SeqIO Task/Mixtures consists of the
following steps:

1.  Choose the model to run inference on.
1.  Choose the SeqIO Task/Mixture to run inference on.
1.  Write a Gin file that configures the model, SeqIO Task/Mixture and other
    details of your inference run.
1.  Launch your experiment locally or on XManager.
1.  Monitor your experiment and access predictions.

These steps are explained in detail in the following sections. An example run
that runs inference on a fine-tuned T5-1.1-Small checkpoint on the
[(Open Domain) (Open Domain) Natural Questions benchmark](https://ai.google.com/research/NaturalQuestions/)
is also showcased.

## Step 1: Choose a model

To run inference on a model, you need a Gin config file that defines the model
params, and the model checkpoint to load from. For this example, a T5-1.1-Small
model fine-tuned on the
[`natural_questions_open_test`](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa/t5_cbqa/tasks.py?l=141&rcl=370261021)
SeqIO Task will be used:

+   Model checkpoint -
    [`cbqa/small_ssm_nq/model.ckpt-1110000`](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/small_ssm_nq/)
+   Model Gin file -
    [`models/t5_1_1_small.gin`](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin).

If you would like to fine-tune your model before inference, please follow the
[fine-tuning](finetune) tutorial, and continue to Step 2.

## Step 3: Write a Gin Config

After choosing the model and SeqIO Task/Mixture for your run, the next step is
to configure your run using Gin. If you're not familiar with Gin, reading the
[T5X Gin Primer](gin.md) is recommended. T5X provides a Gin file that configures
the T5X inference job (located at
[`runs/infer.gin`](https://github.com/google-research/t5x/blob/main/t5x/configs/runs/infer.gin)) to
run inference on SeqIO Task/Mixtures, and expects a few params from you. These
params can be specified in a separate Gin file, or via commandline flags.
Following are the required params:

+   `CHECKPOINT_PATH`: This is the path to the model checkpoint (from Step 1).
    For the example run, set this to
    `'gs://t5-data/pretrained_models/cbqa/small_ssm_nq/model.ckpt-1110000'`.
+   `MIXTURE_OR_TASK_NAME`: This is the SeqIO Task or Mixture name to run
    inference on (from Step 2). For the example run, set this to
    `'natural_questions_open'`.
+   `MIXTURE_OR_TASK_MODULE`: This is the Python module that contains the SeqIO
    Task or Mixture. For the example run, set this to
    `'google_research.t5_closed_book_qa.t5_cbqa.tasks'`.
    Note that this module must be included as a dependency in the T5X inference
    [binary](https://github.com/google-research/t5x/blob/main/t5x/BUILD;l=74;rcl=398627055). Most
    common Task modules, including `t5_closed_book_qa`, are already included. If
    your module is not included, see the
    [Advanced Topics section](#custom-t5x-binaries) at the end of this tutorial
    for instructions to add it.
+   `TASK_FEATURE_LENGTHS`: This is a dict mapping feature key to maximum length
    for that feature. After preprocessing, features are truncated to the
    provided value. For the example run, set this to `{'inputs': 38, 'targets':
    18}`, which is the maximum token length for the test set.
+   `INFER_OUTPUT_DIR`: A path to write inference outputs to. When launching
    using XManager, this path is automatically set and can be accessed from the
    XManager Artifacts page. When running locally using Blaze, you can
    explicitly pass a directory using a flag. Launch commands are provided in
    the next step.

In addition to the above params, you will need to import
[`infer.gin`](https://github.com/google-research/t5x/blob/main/t5x/configs/runs/infer.gin) and the
Gin file for the model, which for the example run is
[`t5_1_1_small.gin`](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin).

```gin
include 'runs/infer.gin'
include 'models/t5_small.gin'
```

Note that the `include` statements use relative paths in this example. You will
pass an appropriate `gin_search_paths` flag to locate these files when launching
your run. Absolute paths to Gin files can also be used, e.g.

```gin
include 't5x/configs/runs/infer.gin'
include 't5x/google/examples/flaxformer_t5/configs/models/t5_1_1_small.gin'
```

Finally, your Gin file should look like this:

```gin
include 'runs/infer.gin'
include 'models/t5_1_1_small.gin'

CHECKPOINT_PATH = 'gs://t5-data/pretrained_models/cbqa/small_ssm_nq/model.ckpt-1110000'
MIXTURE_OR_TASK_NAME = 'closed_book_qa'
MIXTURE_OR_TASK_MODULE = 'google_research.t5_closed_book_qa.t5_cbqa.tasks'
TASK_FEATURE_LENGTHS = {'inputs': 38, 'targets': 18}
```

See
[`t5_1_1_small_cbqa_natural_questions.gin`](https://github.com/google-research/t5x/blob/main/t5x/google/examples/flaxformer_t5/configs/examples/inference/t5_1_1_small_cbqa_natural_questions.gin)
for this example. Make sure that your Gin file is linked as a data dependency to
the T5X inference
[binary](https://github.com/google-research/t5x/blob/main/t5x/BUILD;l=74;rcl=398627055). 

## Step 4: Launch your experiment

To launch your experiment locally (for debugging only; larger checkpoints may
cause issues), run the following on commandline:

```sh
INFER_OUTPUT_DIR="/tmp/model-infer/"
python -m t5x.infer \
  --gin_file=t5x/google/examples/flaxformer_t5/configs/examples/inference/t5_1_1_small_cbqa_natural_questions.gin \
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
  --alsologtostderr
```

Note that multiple comma-separated paths can be passed to the `gin_search_paths`
flag, and these paths should contain all Gin files used or included in your
experiment.


## Step 5: Monitor your experiment and parse results


After inference has completed, you can view predictions in the `jsonl` files in
the output dir. JSON data is written in chunks and combined at the end of the
inference run. Refer to [Sharding](#sharding) and
[Checkpointing](#checkpointing) sections for more details.
