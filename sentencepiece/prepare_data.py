import torch
import argparse


def handle_text_lines_and_write(out, text_to_handle, line_length):
    """ due to the specifics of the VLSP competitions, all the data should be lowercased.
    """
    assert(isinstance(text_to_handle, str))
    assert(isinstance(line_length, int))

    text_to_handle = text_to_handle.lower()
    lines = []
    while len(text_to_handle) > line_length:
        lines.append(text_to_handle[:line_length])
        text_to_handle = text_to_handle[line_length:]

    lines.append(text_to_handle)

    for line in lines:
        out.write(line)
        out.write("\n")


if __name__ == '__main__':
    """
    Every file is expected to have the following JSONL structure, serialized via torch:
    ----------------
    "cluster": {
        "document": [text_1, ... , text_n],
        "summary": "text"
    }
    ----------------
    
    NOTE: for source files, originally we consider the following:
        "vlsp-vmds-vims/wceptest.pt",
        "vlsp-vmds-vims/wceptrain.pt",
        "vlsp-vmds-vims/wcepvalid.pt"
        "newscorpus/news-corpus-step1/newshead/train/" --- 1000 docs.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--line_length", default=1024, type=int)
    parser.add_argument('-n', "--files", nargs='+', default=[])
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    print(args)

    with open(args.output, "w") as out_file:
        # For every file.
        for file in args.files:
            # For every cluster in file.
            for cluster in torch.load(file):
                # Handle texts.
                for text in cluster["document"]:
                    handle_text_lines_and_write(out=out_file,
                                                text_to_handle=text,
                                                line_length=args.line_length)
                # Handle texts summary.
                handle_text_lines_and_write(out=out_file,
                                            text_to_handle=cluster["summary"],
                                            line_length=args.line_length)
