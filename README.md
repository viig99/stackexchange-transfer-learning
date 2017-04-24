# Kaggle Stackexchange Transfer Learning

Adding the code which helped me place in the top 1% of the [kaggle tranfer learning challenge](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags).

It has preprocessing steps to get only relavent information from the text, standardizing it by looking for POS tags, lematizing it.

It then goes uses the [6b glove embeddings](https://nlp.stanford.edu/projects/glove/) for the transfer learning part, then learns the model on a embeddings + cnn + lstm network and uses hamming loss to pick the right cutoffs per label.

I haven't added the glove embeddings or the test, train data since the are above the github threshold, feel free to create a issue in case you are unable to find them.

Happy Kaggling!