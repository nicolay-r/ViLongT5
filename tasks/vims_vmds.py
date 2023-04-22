import functools
from os.path import dirname, join, realpath, exists
import seqio
import t5

from common import INPUT_SIZES
from vocab_default import DEFAULT_VOCAB

from tasks.common import iter_files
from t5.evaluation import metrics


def register(dir_name, model_size, delim="|"):
    """ This task is dedicated for the model pretraining.
    """
    assert(isinstance(delim, str))

    DEFAULT_OUTPUT_FEATURES = {
        "inputs": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
        "targets": seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
    }

    target_dir = join(dirname(realpath(__file__)), dir_name)

    #################################################################
    # This task is dedicated for the finetunning under ViMs and VMDS.
    seqio.TaskRegistry.add(
        "vims_vmds_finetune_{}".format(model_size),
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "train": list(iter_files(target_dir, ends_with="train.pt")),
                "validation": list(iter_files(target_dir, ends_with="valid.pt")),
            },
            skip_header_lines=0),
        preprocessors=[
            functools.partial(t5.data.preprocessors.parse_tsv, field_delim=delim),
            seqio.preprocessors.tokenize,
        ],
        metric_fns=[metrics.rouge],
        output_features=DEFAULT_OUTPUT_FEATURES)

    #################################################################
    # This is task dedicated for testing/evaluating model under VMDS.
    seqio.TaskRegistry.add(
        "vmds_test_{}".format(model_size),
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "test": list(iter_files(target_dir, ends_with="VMDStest.pt"))
            },
            skip_header_lines=0),
        preprocessors=[
            functools.partial(t5.data.preprocessors.parse_tsv, field_delim=delim),
            seqio.preprocessors.tokenize,
        ],
        metric_fns=[metrics.rouge],
        output_features=DEFAULT_OUTPUT_FEATURES)

    #################################################################
    # This is task dedicated for testing/evaluating model under ViMs.
    seqio.TaskRegistry.add(
        "vims_test_{}".format(model_size),
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "test": list(iter_files(target_dir, ends_with="ViMstest.pt"))
            },
            skip_header_lines=0),
        preprocessors=[
            functools.partial(t5.data.preprocessors.parse_tsv, field_delim=delim),
            seqio.preprocessors.tokenize,
        ],
        metric_fns=[metrics.rouge],
        output_features=DEFAULT_OUTPUT_FEATURES)

###################### VIMS-VMDS #######################
# Registering task for tiny and large.

task_and_dirs = [
    ("large", join(dirname(realpath(__file__)), "longt5_vims_vmds_{input}-{output}".format(
        input=INPUT_SIZES["large"]["inputs"], output=INPUT_SIZES["large"]["targets"]))),
]

for model_size, dir_name in task_and_dirs:
    if exists(dir_name):
        register(dir_name=dir_name, model_size=model_size)
