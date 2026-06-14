from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

save_path = 'models/dummy-baseline'
os.makedirs(save_path, exist_ok=True)

# Smallest possible pretrained model as placeholder
model_name = 'google/bert_uncased_L-2_H-128_A-2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print('Saved to', save_path)