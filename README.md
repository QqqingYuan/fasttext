# fasttext

[fasttext](https://github.com/facebookresearch/fastText) with hierarchical softmax, implemented by tensorflow.

The corpus can be find [here](https://github.com/facebookresearch/fastText/blob/master/tutorials/supervised-learning.md).

reference: [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf).

# Requirements

- python 3.4 or newer
- tensorflow 0.12.0rc1

# About the code

The huffman tree should be constructed before training model. 

paths_length.npy   the length of huffman coding of every label.

cooking.train       train file , contained 12404 examples.

cooking.valid       test file , contained 3000 examples.

labels.dict         the dict of labels.

words.dict          the dict of all words.
