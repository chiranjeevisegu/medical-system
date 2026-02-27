from transformers import AutoModel, AutoTokenizer
import torch

model_name = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, use_safetensors=True)

sample_text = "Discharge summary: patient admitted with sepsis and acute kidney injury."
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=256)

with torch.inference_mode():
    outputs = model(**inputs)

print("MODEL_OK")
print("last_hidden_state_shape:", tuple(outputs.last_hidden_state.shape))
