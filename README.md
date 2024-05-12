# Classifying text with DistilBERT and Tensorflow

This notebook is for the [🤗Transformers doc](https://huggingface.co/transformers/notebooks.html#community-notebooks).

Learn how to use powerful transformer models for text classification tasks!

# distilBERT-model-testing for Text Classification 
Notebook example from Hugging Face - A notebook on how to finetune DistilBERT for text classification in TensorFlow. 🌎
https://huggingface.co/docs/transformers/model_doc/distilbert 


Text classification using tokenization involves representing each document as a set of tokens or words. These tokens are often processed to remove stopwords, punctuation, and perform stemming or lemmatization.

For classifying opinions (pos,neg,neutral), the following sentiment analysis models were tested:

- distilbert/distilbert-base-uncased-finetuned-sst-2-english
- michellejieli/inappropriate_text_classifier


Later I will try to fine-tune my own version of BERT (instead of distilBERT) using the same dataset from MichelleJieli's model. Later I will compare and contrast the test error between different sentiment analysis models pretrained on the same dataset.

After the model is fine-tuned, it is possible to save the model using a command in the pytorch/tensorflow library and call it locally instead of referencing a cloud stored model on Hugging Face 
