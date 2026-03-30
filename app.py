import argparse
from datetime import datetime

import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from database.database import Base, SessionLocal, engine
from database.model import Receipt
from preprocessing import label_names


def load_classifier(model_path: str, model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_one(model, tokenizer, device: str, text: str, max_length: int) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=-1).item())
    return label_names[pred_id]


def save_prediction(user_text: str, prediction: str) -> int:
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        row = Receipt(text=user_text, prediction=prediction, created_at=datetime.now())
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def run_gui(model_path: str, model_name: str, max_length: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_classifier(model_path, model_name, device)

    # Renk paleti: temiz/beyaz + soft sarı.
    PANEL_BG = "white"
    BORDER = "#E5E7EB"
    TEXT_DARK = "#0F172A"
    TEXT_MUTED = "#6B7280"
    OUTPUT_BG = "#F8FAFC"
    ACCENT = "#FDE68A"  # soft yellow
    ACCENT_DARK = "#F59E0B"

    root = tk.Tk()
    root.title("Turkish Youth Anxiety - Panel")
    root.configure(bg=PANEL_BG)
    root.geometry("760x540")

    # Kart (modern hissiyat icin hafif border + padding)
    outer = tk.Frame(root, padx=18, pady=18, bg=PANEL_BG)
    outer.pack(fill="both", expand=True)

    title = tk.Label(
        outer,
        text="Düşünce Sınıflandırma",
        bg=PANEL_BG,
        fg=TEXT_DARK,
        font=("Segoe UI", 20, "bold"),
    )
    title.pack(anchor="w", pady=(0, 12))

    input_label = tk.Label(
        outer,
        text="Düşünce Metni",
        bg=PANEL_BG,
        fg=TEXT_MUTED,
        font=("Segoe UI", 11, "bold"),
    )
    input_label.pack(anchor="w")

    input_box = ScrolledText(
        outer,
        width=85,
        height=10,
        bg="white",
        fg=TEXT_DARK,
        insertbackground=TEXT_DARK,
        relief="solid",
        bd=1,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    input_box.pack(fill="x", pady=(6, 14))

    output_var = tk.StringVar(value="Tahmin burada görünecek.")
    output_frame = tk.Frame(outer, bg=OUTPUT_BG, bd=1, relief="solid")
    output_frame.pack(fill="x", pady=(0, 14))

    output_label = tk.Label(
        output_frame,
        textvariable=output_var,
        bg=OUTPUT_BG,
        fg=TEXT_DARK,
        font=("Segoe UI", 12, "bold"),
        justify="left",
        wraplength=650,
        padx=12,
        pady=10,
    )
    output_label.pack(anchor="w")

    controls = tk.Frame(outer, bg=PANEL_BG)
    controls.pack(anchor="w", fill="x")

    def on_predict_and_save():
        user_text = input_box.get("1.0", "end").strip()
        if not user_text:
            messagebox.showwarning("Uyarı", "Lütfen bir metin girin.")
            return

        try:
            prediction = predict_one(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=user_text,
                max_length=max_length,
            )
            pred_id = save_prediction(user_text=user_text, prediction=prediction)
            output_var.set(f"Tahmin: {prediction}  |  ID: {pred_id}")
        except Exception as e:
            messagebox.showerror("Hata", f"İşlem başarısız: {e}")

    def on_clear():
        input_box.delete("1.0", "end")
        output_var.set("Tahmin burada görünecek.")

    predict_btn = tk.Button(
        controls,
        text="Sınıflandır ve Kaydet",
        command=on_predict_and_save,
        bg=ACCENT,
        fg=TEXT_DARK,
        padx=14,
        pady=10,
        activebackground=ACCENT_DARK,
        activeforeground=TEXT_DARK,
        font=("Segoe UI", 11, "bold"),
        relief="flat",
    )
    predict_btn.pack(side="left")

    clear_btn = tk.Button(
        controls,
        text="Temizle",
        command=on_clear,
        bg=PANEL_BG,
        fg=TEXT_DARK,
        padx=14,
        pady=10,
        activebackground=BORDER,
        font=("Segoe UI", 11),
        relief="groove",
        bd=1,
    )
    clear_btn.pack(side="left", padx=(12, 0))

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Düşünce paneli ve PostgreSQL kaydı")
    parser.add_argument("--model-path", default="./results/checkpoint-327")
    parser.add_argument("--model-name", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    run_gui(
        model_path=args.model_path,
        model_name=args.model_name,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
