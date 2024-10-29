pip install transformers
pip install peft
pip install torch
pip install pandas
pip install scikit-learn

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Dataset
data_path = '/Users/pratyushsinghal/Downloads'
df = pd.read_excel(data_path)

data = df.iloc[:, 0].tolist()
prompts = []
responses = []

for entry in data:
    if "[INST]" in entry and "</s>" in entry:
        inst_split = entry.split("[/INST]")
        prompt = inst_split[0].replace("<s>[INST]", "").strip()
        response = inst_split[1].replace("</s>", "").strip()
        prompts.append(prompt)
        responses.append(response)

train_prompts, test_prompts, train_responses, test_responses = train_test_split(
    prompts, responses, test_size=0.2, random_state=42
)

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

def tokenize_function(prompts, responses):
    encodings = tokenizer(
        prompts, text_target=responses, 
        max_length=300, truncation=True, padding="max_length"
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

train_encodings = tokenize_function(train_prompts, train_responses)
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_encodings['labels'])
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=8,
    learning_rate=0.0001,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)
trainer.train()

model.save_pretrained("./fine_tuned_llama2_security")
tokenizer.save_pretrained("./fine_tuned_llama2_security")

model = AutoModelForCausalLM.from_pretrained("./fine_tuned_llama2_security")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_llama2_security")

def generate_response(prompt):
    input_text = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

predicted_labels = []
true_labels = []

for i, prompt in enumerate(test_prompts):
    generated_response = generate_response(prompt)
    
    if "I can't provide you with this information" in generated_response:
        predicted_labels.append(1)  
    else:
        predicted_labels.append(0)
    
    true_label = 1 if "I can't provide you with this information" in test_responses[i] else 0
    true_labels.append(true_label)

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print("Evaluation Metrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"Accuracy: {accuracy:.3f}")
