
# Turkish Youth Anxiety Classification

This project classifies Turkish youth-anxiety related texts into **categories (thema)** using a BERT-based multi-class model.  
It also:
- **Visualizes performance metrics** (`visualization.py`)
- **Classifies user input from a panel and saves it to PostgreSQL** (`app.py`)

## Project structure

- `preprocessing.py`: Reads the Excel file, label-encodes classes, and creates a train/test split (`X_train`, `X_test`, `y_train`, `y_test`, `label_names`)
- `train.py`: Fine-tunes the model using HuggingFace `Trainer`
- `test.py`: Produces a confusion matrix + classification report
- `visualization.py`: Produces per-category F1 / accuracy charts + confusion matrix
- `database/`
  - `database.py`: SQLAlchemy engine + session
  - `model.py`: PostgreSQL table model (`predictions`)
- `app.py`: Modern panel (Tkinter). Text → prediction → PostgreSQL insert

## Requirements

- Python 3.10+ ( currently using 3.13)
- PostgreSQL (localhost:5432)
- (Recommended) Virtual environment: `venv` / `conda`

Python packages (typical):
- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `openpyxl`
- `matplotlib`
- `seaborn`
- `sqlalchemy`
- `psycopg2-binary`

## Setup

Virtual environment (optional, recommended):

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -U pip
pip install torch transformers scikit-learn pandas openpyxl matplotlib seaborn sqlalchemy psycopg2-binary
```

## Data

Data source: `dataset.xlsx` (Sheet: `Sheet1`)

`preprocessing.py`:
- Encodes `thema` → `thema1`
- Uses `contents` as the text input column

## Training

```bash
python train.py
```

Training outputs are written to `./results/` (e.g. `results/checkpoint-327`).

## Test (report + confusion matrix)

```bash
python test.py
```

## Visualization (per-category F1/accuracy + confusion matrix)

Default checkpoint:

```bash
python visualization.py
```

With a different checkpoint:

```bash
python visualization.py --model-path "./results/checkpoint-327" --max-length 128
```

Outputs:
- `results/figures/f1_score_per_category.png`
- `results/figures/accuracy_per_category_ovr.png`
- `results/figures/confusion_matrix.png`

## PostgreSQL setup

Connection string is in `database/database.py`:

```text
postgresql://postgres:mit.2021A@localhost:5432/turkish_youth_anx_clas
```

The table you created in pgAdmin (expected schema):

```sql
CREATE TABLE predictions (
  id SERIAL PRIMARY KEY,
  text TEXT,
  prediction TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

> Note: `app.py` inserts rows into the same table (`predictions`) using SQLAlchemy.

## Panel (type text → classify → save to PostgreSQL)

```bash
python app.py
```

With parameters:

```bash
python app.py --model-path "./results/checkpoint-327" --max-length 128
```

Panel flow:
- Type/paste the text
- Click **Sınıflandır ve Kaydet**
- The predicted category and the inserted row ID are shown in the UI

## Common issues

### “database does not exist”
Make sure the database name in `DATABASE_URL` actually exists in PostgreSQL.

### Model checkpoint bulunamadı
Run `train.py` first or provide the correct path via `--model-path`.

## License

No license has been specified for this repository.

