import os
import argparse
import torch
from tqdm import tqdm
from os.path import join

from tasks.common import crop_and_pad_input_string, SENT_SEP_TOKEN, DOC_SEP_TOKEN

if __name__ == '__main__':
    """ Artificial summary former based on cluseter of the documents, with
        the pre-cacluated scores for every sentences. Every cluster represent 
        a set of ducuments, and this script provides an extractive summary 
        for every cluster of the document.
        
        from input, where each document represent 
        [
            ...
            "data": [   # cluster
                [ ... ] # doc1.
                [ ... ] # doc2.
                [ ... ] # doc3.
                [ ... ] # ...
                [
                    {"text": "any sentence text goes here", "pyramid_rouge": 0.4}
                    {"text": "text goes here", "pyramid_rouge": 0.1}
                    {"text": "text goes", "pyramid_rouge": 0.3}
                    {"text": "text goes here", "pyramid_rouge": 0.2}
                ]
            ]
        ]
        
        -> 
        
        TSV format, which contains input and output of entries.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_length_input", default=4090, type=int)
    parser.add_argument("--max_length_output", default=1020, type=int)
    parser.add_argument('--separator', default='|', type=str)
    parser.add_argument("--m", default=5, type=int,
                        help="amount of sentences per summary. "
                             "Note: logic even complex, see implementation in code.")
    parser.add_argument('--sep_to_replace', default='_', type=str)
    parser.add_argument('--limit', default=5, type=int)
    parser.add_argument('--percentage_crop', default=0.8, type=float)
    parser.add_argument('--source_dir', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--metric', default='pyramid_rouge', type=str,
                        help="metric utilized to consider the level of sentence salient for summarization.")

    args = parser.parse_args()

    print(args)

    assert(args.limit > 0 or args.limit == -1)

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
            scored_data = torch.load(target)

            for item in scored_data:

                if not isinstance(item, dict):
                    print("Skipped. Type item={}".format(type(item)))
                    continue

                # For all the articles within a single cluster.
                # Every cluster consist of multiple-documents, which
                # were artifically splitted into parts.
                for key, article_value in item.items():

                    scores = []
                    texts = []

                    if not key == "data":
                        continue

                    if not isinstance(article_value, list):
                        continue

                    # For each artificial article.
                    for texts_list in article_value:
                        for text in texts_list:
                            text_added = False
                            for text_key, text_value in text.items():
                                if text_key == args.metric:
                                    text_id = len(texts) if not text_added else len(texts) - 1
                                    scores.append((text_id, text_value))
                                if text_key == "text":
                                    text_added = True
                                    texts.append(text_value)

                            # We artifically put a SENT_SEP token.
                            texts.append(SENT_SEP_TOKEN)

                        # We artifically put a DOC_SEP separator.
                        texts.append(DOC_SEP_TOKEN)

                    # Considering a half part of the text for summary
                    # and at leas minimal amount of sentences proposed by `m`.
                    actual_m = min(args.m, int(len(texts) / 2))

                    source = ' '.join(texts)
                    top_scores = list(reversed(sorted(scores, key=lambda item: item[1])))[:actual_m]
                    summary_t_inds = set([ind for ind, score in top_scores])
                    summary = ' '.join([texts[t_ind] for t_ind, t in enumerate(texts) if t_ind in summary_t_inds])

                    # Formatting: cropping data, replacing the sep char with the other one.
                    source = crop_and_pad_input_string(text=source, length=args.max_length_input,
                                                       percentage_crop=args.percentage_crop, with_eos=True)
                    summary = crop_and_pad_input_string(text=summary, length=args.max_length_output,
                                                        percentage_crop=args.percentage_crop, with_eos=True)
                    source = source.replace(args.separator, args.sep_to_replace)
                    summary = summary.replace(args.separator, args.sep_to_replace)

                    # Writing down the row.
                    out_file.write("{}\n".format(args.separator.join([source, summary])))
