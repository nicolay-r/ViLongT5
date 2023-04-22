import os
from os.path import join


SENT_SEP_TOKEN = "<sent-sep>"
DOC_SEP_TOKEN = "<doc-sep>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

INPUT_SIZES = {
    "large": {"inputs": 2048, "targets": 512},
}


def crop_and_pad_input_string(text, length, percentage_crop, with_eos, provide_padding=False):
    assert(isinstance(text, str))
    assert(isinstance(percentage_crop, float) and 0 < percentage_crop <= 1)
    assert(isinstance(length, int) and length > 0)
    assert(isinstance(with_eos, bool))

    # lowercase the contents.
    # Due to the VLSP organization specifics.
    text = text.lower()

    # Actual length in contents, excluding <eos>.
    # Reason is that during the model tokenization, some tokens
    # might be departed and therefore the whole sequence is expected
    # to be even longer. Since we would like to consider this sequence
    # as a list of documents, we also would like to place <eos> and
    # for some cases see that <eos> will be a part of the result input
    # sequence.
    actual_tokens_length = int(length * percentage_crop)

    tokens = text.split()
    # Remove empty tokens
    tokens = [token.strip() for token in tokens if token.strip()]

    if len(tokens) > actual_tokens_length:
        tokens = tokens[:actual_tokens_length]

    if with_eos:
        tokens += [EOS_TOKEN]

    # Provide padding.
    if provide_padding:
        tokens += [PAD_TOKEN] * (length - len(tokens))
        assert(len(tokens) == length)

    return " ".join(tokens)


def iter_files(dir, ends_with):

    for file in os.listdir(dir):
        filename = os.fsdecode(file)

        ok = False
        if isinstance(ends_with, str) and filename.endswith(ends_with):
            ok = True

        # Switch to the next.
        if not ok:
            continue

        yield join(dir, filename)
