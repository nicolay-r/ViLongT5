from os.path import dirname, join, realpath

import seqio


def setup_vocabulary():
    return DEFAULT_VOCAB


DEFAULT_EXTRA_IDS = 100
current_dir = join(dirname(realpath(__file__)))
DEFAULT_VOCAB = seqio.SentencePieceVocabulary(
    join(current_dir, "sentencepiece/model/vietnam.model"), DEFAULT_EXTRA_IDS)