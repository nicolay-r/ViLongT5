# ViLongT5
[![PRs welcome!](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()

A pretrained Transformer-based encoder-decoder model for the
multi-document text-summarization
problem in Vietnamese language. 
With [LongT5](https://github.com/google/flaxformer) original architecture 
implementation, 
ViLongT5 is trained on a large NewsCorpus of news Vietnamese texts.
We benchmark LongViT5 on multidocument text-summarization tasks, 
Abstractive Text Summarization and Named Entity Recognition. 
All the experiments are shown in our paper 
**[Pre-training LongT5 for Vietnamese Mass-Media
Multi-document Summarization Task]()**


## Pretrained Models
**Vocabulary:**
[LongViT5_vocab]()

Model        | Gin File Location                                                                  | Checkpoint Location|
------------ | ---------------------------------------------------------------------------------- | -------------------|
ViT5-Large | [LongViT5_large.gin]() | [storage-path]() | [ViT5-Large-1024 (1.5M)](https://huggingface.co/VietAI/vit5-large)

## Finetunning

📄 Example with Flaxformer: [to-be-added]()


## Results

> TODO. Image with the results.

### Datasets
- [NewsCorpus]()
- [VMDS]()
- [VNDS]()
- [ViMS]()

## Citation
```
@inproceedings{rusnachenko2023pretraining,
    title = "Pre-training {LongT5} for Vietnamese Mass-Media Multi-document Summarization Task",
    author = "Nicolay, Rusnachenko and The Anh, Le and Ngoc Diep, Nguyen",
    booktitle = "Proceedings of Artificial Intelligence and Natural Language",
    year = "2023"
}
```