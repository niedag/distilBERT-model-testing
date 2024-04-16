from transformers import pipeline, AutoModelForSequenceClassification
import pprint

classifier = pipeline("sentiment-analysis", model="michellejieli/inappropriate_text_classifier")

data = [
    "You're so naive Roxy. You see the good in everybody, even when it's not there. You're living a fantasy. "
    "There is no Easter Bunny, there is no Tooth Fairy, and there is no Queen of England."
    "This is the real world, and you need to wake up! You dare challenge Megamind? This town isn't big enough for two supervillains."
    "Oh, you're a villain alright, just not a super one. Yeah? What's the difference? Presentation!"
    # "I love my country so much",
    # "God bless America",
    # "I feel so good about the changes I'm making!",
    # "I think Trump should tear his buildings down for agricultural space",
    # "I think Trump should tear his buildings down for agricultural space while he's inside",
]
# Initial problem with loading this script due to lack of tf_model.h5 file which stores weights from TensorFlow
pprint.pp(classifier(data))
