from transformers import pipeline
import pprint as pp

basic_sentiment_pipline = pipeline('sentiment-analysis')
data = [
    "I love you",
    "I hate you"
]
print(basic_sentiment_pipline(data))

# Model: 3 https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
# This file runs the base distilbert model 