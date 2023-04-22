import argparse
import os
from os.path import join

import torch
from tqdm import tqdm

from tasks.common import SENT_SEP_TOKEN, DOC_SEP_TOKEN, crop_and_pad_input_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_length_input", default=4090, type=int)
    parser.add_argument("--max_length_output", default=1020, type=int)
    parser.add_argument('--separator', default='|', type=str)
    parser.add_argument('--sep_to_replace', default='_', type=str)
    parser.add_argument('--limit', default=5, type=int)
    parser.add_argument('--percentage_crop', default=0.8, type=float)
    parser.add_argument('--source_dir', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)

    args = parser.parse_args()

    print(args)

    assert (args.limit > 0 or args.limit == -1)

    # Create output dir.
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    cur_limit = 0

    for file in tqdm(os.listdir(args.source_dir)):
        filename = os.fsdecode(file)

        if not filename.endswith(".pt"):
            continue

        # break by a given limit
        cur_limit += 1
        if args.limit != -1 and args.limit == cur_limit:
            break

        with open(join(args.output_dir, filename), "w") as out_file:

            target = join(args.source_dir, filename)
            clusters = torch.load(target)

            for cluster in clusters:

                # For all the articles within a single cluster.
                # Every cluster consist of multiple-documents, which
                # were artifically splitted into parts.
                texts = []
                summary = None
                for key, data in cluster.items():

                    # For each artificial article.
                    if key == "document":
                        for document in data:
                            for sentence in document.split('\n'):
                                texts.append(sentence)
                                # Do sentence separation.
                                texts.append(SENT_SEP_TOKEN)
                            # We artifically put a DOC_SEP separator.
                            texts.append(DOC_SEP_TOKEN)

                    if key == "summary":
                        summary = data

                # Formatting: cropping data, replacing the sep char with the other one.
                source = ' '.join(texts)
                source = crop_and_pad_input_string(text=source, length=args.max_length_input,
                                                   percentage_crop=args.percentage_crop, with_eos=True)
                summary = crop_and_pad_input_string(text=summary, length=args.max_length_output,
                                                    percentage_crop=args.percentage_crop, with_eos=True)
                source = source.replace(args.separator, args.sep_to_replace)
                summary = summary.replace(args.separator, args.sep_to_replace)

                # Writing down the row.
                out_file.write("{}\n".format(args.separator.join([source, summary])))
