# Training SentencePiece Model

The pre-trained version of the model is provided at `model` directory.
This README represents the tutorial of the manual application of the `sentencepiece`
towards the manually prepared texts.

There are two steps expected to be accomplished:

1. Install sentencepiece library first:
```
pip install sentencepiece
```

2. Then, there is a need to prepare a texts, composed 
in a form of the text file.
To train model, based on the `data.txt` contents, please refer to the following command:
```
python train_model.py --input data.txt
```

> **NOTE**: In terms of the original data text file preparation, please follow the `prepare_data.py` script for a greater details.