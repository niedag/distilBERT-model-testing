from transformers import pipeline, AutoModelForSequenceClassification


classifier = pipeline("sentiment-analysis", model="michellejieli/inappropriate_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("michellejieli/inappropriate_text_classifier", from_pt=True)
#print(classifier("I see you’ve set aside this special time to humiliate yourself in public."))

# Initial problem with loading this script due to lack of tf_model.h5 file which stores weights from TensorFlow
print(model("I see you’ve set aside this special time to humiliate yourself in public."))
