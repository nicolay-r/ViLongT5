import sentencepiece as spm
from tasks.common import PAD_TOKEN, EOS_TOKEN, DOC_SEP_TOKEN, SENT_SEP_TOKEN, INPUT_SIZES

spm.SentencePieceTrainer.train('--input=all.txt' +
                               '--model_prefix=vietnam ' +
                               '--vocab_size=32000 ' +
                               '--train_extremely_large_corpus ' +
                               # We considering the input length exact the same as for the LARGE LongT5 model.
                               '--max_sentence_length={} '.format(str(INPUT_SIZES["large"]["inputs"])) +
                               '--user_defined_symbols={}'.format(",".join([SENT_SEP_TOKEN,
                                                                            DOC_SEP_TOKEN,
                                                                            EOS_TOKEN,
                                                                            PAD_TOKEN])))
