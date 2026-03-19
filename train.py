
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from preprocessing import X_train, X_test, y_train, y_test

# =============================================================================
# 1. MODEL & TOKENIZER CONFIGURATION
# =============================================================================
# Using BERTurk (Turkish BERT) base model for text classification
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(texts):
    """
    Tokenizes input text sequences and prepares them for BERT model input.
    - padding: Ensures all sequences in a batch have the same length.
    - truncation: Clips sequences exceeding the max_length.
    - return_tensors="pt": Returns PyTorch tensors.
    """
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

# =============================================================================
# 2. PYTORCH DATASET WRAPPER
# =============================================================================
class AnxietyDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class required by the Hugging Face Trainer API.
    Maps input encodings and target labels to a dictionary format.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Using clone().detach() to avoid unnecessary computation graphs and warnings
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Tokenize data splits
train_encodings = tokenize_function(X_train)
test_encodings = tokenize_function(X_test)

# Initialize dataset objects
train_dataset = AnxietyDataset(train_encodings, y_train)
test_dataset = AnxietyDataset(test_encodings, y_test)

# =============================================================================
# 3. MODEL INITIALIZATION
# =============================================================================
# Initializing BERT with 6 output labels for multi-class anxiety classification
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6
)

# =============================================================================
# 4. TRAINING ARGUMENTS & METRICS
# =============================================================================
training_args = TrainingArguments(
    output_dir='./results',          # Path for storing model checkpoints
    num_train_epochs=3,              # Total number of training iterations
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Path for storing logs
    eval_strategy="epoch",           # Evaluation frequency
    save_strategy="no",              # Disable checkpoint saving to save disk space
    save_total_limit=1               # Maximum number of checkpoints to keep
)

def compute_metrics(eval_pred):
    """
    Evaluation metric function to calculate accuracy during training.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# =============================================================================
# 5. TRAINING EXECUTION
# =============================================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print("Execution started: Fine-tuning BERT for Turkish Anxiety Classification...")
    trainer.train()

# Örnek bir tahmin denemesi
    test_sentence = "Gelecekte iş bulabilecek miyim diye çok kaygılanıyorum."
    inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=256)


# Modeli değerlendirme moduna al
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
print(f"Cümle: {test_sentence}")
print(f"Tahmin Edilen Kategori ID: {prediction.item()}")
