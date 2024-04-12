from transformers import pipeline, AutoModelForSequenceClassification


classifier = pipeline("sentiment-analysis", model="michellejieli/inappropriate_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("michellejieli/inappropriate_text_classifier")

# Initial problem with loading this script due to lack of tf_model.h5 file which stores weights from TensorFlow
print(classifier("Who handles the data at rest"))

