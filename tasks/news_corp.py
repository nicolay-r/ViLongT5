import functools
import sys

import seqio
import t5.data

from t5.evaluation import metrics
from vocab_default import DEFAULT_VOCAB
from common import INPUT_SIZES
from os.path import dirname, realpath, join, exists, basename

from tasks.common import iter_files


def register(main_dir, task_name, model_size, delim="|"):
    """ This task is dedicated for the model pretraining.
    """
    assert(isinstance(model_size, str))

    # Amount of documents dedicated for
    # the validation and testing.

    test_val_part_size = 240

    total_list = sorted(list(iter_files(main_dir, ends_with='.pt')))
    rest_part = total_list[-test_val_part_size:]
    half = int(len(rest_part) / 2)

    # declaring parts: train, validation, test.
    train_part = total_list[:-test_val_part_size]
    validation_part = rest_part[:half]
    test_part = rest_part[half:]

    # logging the related information.
    print("-----------------------------------------")
    print("For {} task:".format(model_size))
    print("-----------------------------------------")
    print("TRAIN:", [basename(f) for f in train_part])
    print("len(TRAIN):", len(train_part))
    print("-----------------------------------------")
    print("VALIDATION:", [basename(f) for f in validation_part])
    print("len(VALIDATION):", len(validation_part))
    print("-----------------------------------------")
    print("TEST:", [basename(f) for f in test_part])
    print("len(TEST):", len(test_part))
    print("-----------------------------------------")
    sys.stdout.flush()

    datasource = seqio.TextLineDataSource(
        split_to_filepattern={
            "train": train_part,
            "validation": validation_part,
            "test": test_part
        },
        skip_header_lines=0)

    seqio.TaskRegistry.add(
        "_".join([task_name, model_size]),
        source=datasource,
        preprocessors=[
            functools.partial(t5.data.preprocessors.parse_tsv, field_delim=delim),
            seqio.preprocessors.tokenize,
        ],
        metric_fns=[metrics.rouge],
        output_features={
            "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
            "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
    })


###################### NEWS-CORPUS #######################
# Registering task for tiny and large.

task_and_dirs = [
    # LARGE
    ("large", join(dirname(realpath(__file__)), "longt5_pretrain_{input}-{output}-train".format(
        input=INPUT_SIZES["large"]["inputs"], output=INPUT_SIZES["large"]["targets"]))),
]

for model_size, train_dir_name in task_and_dirs:
    if exists(train_dir_name):
        register(main_dir=train_dir_name, task_name="newscorp", model_size=model_size)
