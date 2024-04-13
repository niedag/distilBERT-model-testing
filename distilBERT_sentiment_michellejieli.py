from transformers import pipeline, AutoModelForSequenceClassification

#model = AutoModelForSequenceClassification.from_pretrained("michellejieli/inappropriate_text_classifier")
classifier = pipeline("sentiment-analysis", model="michellejieli/inappropriate_text_classifier")

# Initial problem with loading this script due to lack of tf_model.h5 file which stores weights from TensorFlow
print(classifier("I see you’ve set aside this special time to humiliate yourself in public."))
print(classifier("I feel so good about the changes I'm making!"))
