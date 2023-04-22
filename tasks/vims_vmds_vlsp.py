import functools
from os.path import join, dirname, realpath, exists

import seqio
import t5

from common import INPUT_SIZES
from tasks.common import iter_files
from vocab_default import DEFAULT_VOCAB
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
    # This is task dedicated for testing/evaluating model under ViMs.
    seqio.TaskRegistry.add(
        "vims_vmds_vlsp_{}".format(model_size),
        source=seqio.TextLineDataSource(
            split_to_filepattern={
                "test": list(iter_files(target_dir, ends_with="wceptest.pt")),
                "train": list(iter_files(target_dir, ends_with="wceptrain.pt")),
                "validation": list(iter_files(target_dir, ends_with="wcepvalid.pt"))
            },
            skip_header_lines=0),
        preprocessors=[
            functools.partial(t5.data.preprocessors.parse_tsv, field_delim=delim),
            seqio.preprocessors.tokenize,
        ],
        metric_fns=[metrics.rouge],
        output_features=DEFAULT_OUTPUT_FEATURES)


###################### VIMS-VMDS-VLSP #######################
# Registering task for tiny and large.

task_and_dirs = [
    ("large", join(dirname(realpath(__file__)), "longt5_vlsp_vmds_vims_{input}-{output}".format(
     input=INPUT_SIZES["large"]["inputs"], output=INPUT_SIZES["large"]["targets"]))),
]

for model_size, dir_name in task_and_dirs:
    if exists(dir_name):
        register(dir_name=dir_name, model_size=model_size)
