import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from preprocessing import le_thema


model_path = "./results/checkpoint-327"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")



def predict_anxiety(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)


    category_name = le_thema.inverse_transform([prediction.item()])[0]
    return category_name



cumle = "Gelecekte iş bulabilecek miyim diye çok kaygılanıyorum."
sonuc = predict_anxiety(cumle)

print(f"\nCümle: {cumle}")
print(f"Modelin Tahmini: {sonuc}")
