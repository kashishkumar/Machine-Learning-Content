from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"], padding=True, truncation=True, return_tensors="pt")
print(inputs)

model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # torch.Size([2, 9, 768])


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)


predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Convert the predicted logits to actual class predictions using model.config.id2label (not model.config.label2id, which is the default label mapping)
label_ids = torch.argmax(predictions, dim=-1)
labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]



