# https://colab.research.google.com/drive/1t-NJadXsPTDT6EWIR0PRzpn5o8oMHzp3?usp=sharing#scrollTo=bxWHZyRQa2fE
import torch
import pprint
import numpy as np
# print(torch.cuda.is_available()) # True - for me at least

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification

def preprocess_function(examples): # For preparing the text inputs for the model
    return tokenizer(examples["text"],truncation = True)

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)

imdb = load_dataset("imdb") # Loads from Hugging Face - large database of movie reviews

# Create a smaller training dataset for faster training times
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

pprint.pp(small_train_dataset[0])
pprint.pp(small_test_dataset[0])

# Creating Distilbert Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Text Pre-processing
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched = True)

# Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

print(data_collator.return_tensors)

# MODEL TRAINING!!!

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)
print(model)