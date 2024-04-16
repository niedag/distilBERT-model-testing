
import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pickle

x = [
     'Great customer service! The food was delicious! Definitely a come again.',
     'The VEGAN options are super fire!!! And the plates come in big portions. Very pleased with this spot, I\'ll definitely be ordering again',
     'Come on, this place is family owned and operated, they are super friendly, the tacos are bomb.',
     'This is such a great restaurant. Multiple times during days that we don\'t want to cook, we\'ve done takeout here and it\'s been amazing. It\'s fast and delicious.',
     'Staff is really nice. Food is way better than average. Good cost benefit.',
     'pricing for this, while relatively inexpensive for a Las Vegas attraction, is completely over the top.',
     'At such a *fine* institution, I find the lack of knowledge and respect for the art appalling',
     'If I could give one star I would...I walked out before my food arrived the customer service was horrible!',
     'Wow the slowest drive thru I\'ve ever been at WOWWWW. Horrible I won\'t be coming back here ever again',
     'Service: 1 out of 5 stars. They will mess up your order, not have it ready after 30 mins calling them before. Worst ran family business Ive ever seen.'
]

y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 20

review = x[0]

tkzr = DistilBertTokenizer.from_pretrained(MODEL_NAME)

inputs = tkzr(review, max_length=MAX_LEN, truncation=True, padding=True)

print(f'review: \'{review}\'')
print(f'input ids: {inputs["input_ids"]}')
print(f'attention mask: {inputs["attention_mask"]}')