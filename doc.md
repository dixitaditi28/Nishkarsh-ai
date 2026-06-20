# Nishkarsh AI — Full Project Documentation

> **Purpose of this document:** A complete, step-by-step log of every action taken, every decision made, every error encountered, and how it was resolved. Written so that anyone (including the original developer 6 months later) can understand exactly what was done and why.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Name Decision](#2-name-decision)
3. [System Specifications](#3-system-specifications)
4. [Four-Day Roadmap](#4-four-day-roadmap)
5. [Day 1 — Detailed Step Log](#5-day-1--detailed-step-log)
   - [Step 1: Project Folder Structure](#step-1-project-folder-structure)
   - [Step 2: Conda Virtual Environment](#step-2-conda-virtual-environment)
   - [Step 3: Dataset Download](#step-3-dataset-download)
   - [Step 4: Jupyter Notebook & EDA](#step-4-jupyter-notebook--eda)
   - [Step 5: Category Mapping](#step-5-category-mapping)
   - [Step 6: Clean Dataset Save](#step-6-clean-dataset-save)
6. [Hurdles & Resolutions](#6-hurdles--resolutions)
7. [Tool Decisions — Why Each Was Chosen](#7-tool-decisions--why-each-was-chosen)
8. [Dataset Rationale — Why US Data for an India Project](#8-dataset-rationale--why-us-data-for-an-india-project)
9. [Data Findings from EDA](#9-data-findings-from-eda)
10. [Current File State](#10-current-file-state)

---

## 1. Project Overview

**Nishkarsh AI** is a complaint intelligence pipeline built for the Indian banking regulatory context.

### The Problem
When a consumer files a complaint with the RBI Ombudsman or on the CMS portal, it lands in a queue with no intelligent routing. Banks manually classify complaints by type, and resolution times vary wildly — ATM fraud complaints sitting next to KYC paperwork disputes. The RBI Ombudsman receives complaints across 15+ categories — ATM/Debit Cards, Mobile/Electronic Banking, Credit Cards, Loans & Advances, Deposit Accounts, Remittances, Recovery Agents, and more — all handled with minimal automation.

There is no system that predicts:
- Which category a complaint belongs to
- How urgent it is
- Whether it's likely to be resolved in favour of the consumer

Small-town and first-time complainants write vague, unstructured text — the system fails them first.

### The Solution
A complaint intelligence pipeline that takes raw unstructured complaint text as input and outputs:
- **Category classification** — one of 15 RBI Ombudsman categories
- **Urgency score** — 1 to 10 based on monetary amount, complaint type, repeat filer status
- **Predicted resolution outcome** — resolved in complainant's favour or not
- **Predicted resolution timeline** — number of days
- **Structured officer brief** — LLM-generated summary for the reviewing officer

---

## 2. Name Decision

Two names were considered:

| | Nishkarsh AI ✅ | FinGuard AI ❌ |
|---|---|---|
| **Origin** | Sanskrit (निष्कर्ष) — means conclusion/outcome | English portmanteau |
| **Relevance** | Directly maps to the core function: predicting outcomes | Sounds like a generic security product |
| **Uniqueness** | Distinctive; not used in the fintech space | Dozens of tools use the Fin+Guard pattern |
| **Context fit** | Works naturally in Indian regulatory context | Could be from any country |

**Decision:** Nishkarsh AI. The name is memorable precisely because it is not English, it is culturally grounded, and it describes the product's purpose exactly — predicting conclusions/outcomes of complaints.

---

## 3. System Specifications

| Component | Detail |
|-----------|--------|
| Machine | Lenovo IdeaPad Gaming Laptop |
| OS | Windows 11 |
| Terminal | PowerShell (in VS Code) |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MB (4GB) |
| CUDA Version | 12.5 (driver) |
| CPU | Intel (with integrated Intel Iris Xe graphics) |
| Python | 3.11 via Anaconda |
| IDE | Visual Studio Code |
| Conda Env Location | D:\conda-envs\nishkarsh |
| PyTorch Build | cu121 (CUDA 12.1 — compatible with 12.5 driver) |

**Important note on drive layout:**
- C: drive had only 2.09GB free — not enough for PyTorch (~3GB)
- D: drive had 244GB free
- Conda environments and package cache were moved to D: permanently (see Hurdles section)

---

## 4. Four-Day Roadmap

The project is structured into 4 focused build days:

### Day 1 — Data Pipeline & ML Foundations
- Download and explore the CFPB consumer complaint dataset
- Map US product categories to 15 RBI Ombudsman categories
- Tokenise complaint text using DistilBERT tokenizer
- Fine-tune DistilBERT for multi-label classification
- Target: 92%+ accuracy baseline

### Day 2 — Regression Heads, Outcome Model & Voice
- Add regression head on top of DistilBERT for resolution time prediction (target MAE ≤2 days)
- Train binary classifier for outcome prediction (favour / not in favour)
- Integrate OpenAI Whisper ASR for Hindi + English voice complaints
- End-to-end pipeline: audio → transcript → classify (target <3 seconds)
- Gemini API summarizer: raw complaint → structured officer brief
- Generate synthetic Indian complaint narratives for domain adaptation

### Day 3 — FastAPI Backend & PostgreSQL
- Design complaint schema in PostgreSQL
- Build REST API endpoints: POST /complaint, GET /queue, GET /stats
- Serve all models via API
- Implement urgency scoring logic (1–10)
- Dockerise the entire stack

### Day 4 — Streamlit Dashboard & AWS Deployment
- Build officer-facing Streamlit dashboard
- Urgency heatmap, week-over-week trend charts, exportable reports
- Deploy Docker containers to AWS EC2
- Host models and data on S3
- Final live demo and metric validation

---

## 5. Day 1 — Detailed Step Log

### Step 1: Project Folder Structure

**What we did:**
Created the root project directory and all subdirectories.

**Commands run (PowerShell):**
```powershell
mkdir nishkarsh-ai
cd nishkarsh-ai
mkdir data, notebooks, models, src, data\raw, data\processed
code .
```

**Why this structure:**
- `data/raw/` — original downloaded files, never modified
- `data/processed/` — cleaned and transformed files ready for training
- `models/` — saved model checkpoints and weights
- `notebooks/` — Jupyter notebooks for exploration and prototyping
- `src/` — production Python scripts (later: FastAPI, utilities, mapping)

Separating raw from processed data is a best practice: if anything goes wrong during processing, the original file is always intact.

**Hurdle encountered:**
The bash command `mkdir data notebooks models src` does not work in PowerShell. PowerShell's `mkdir` only accepts multiple folders via comma-separated syntax: `mkdir data, notebooks, models, src`.

**Resolution:** Used PowerShell-correct syntax with commas.

---

### Step 2: Conda Virtual Environment

**What we did:**
Created an isolated Python 3.11 environment on D: drive and installed all required ML libraries.

**Why Python 3.11 specifically:**
- Python 3.10 (originally planned) is slightly older but fully supported
- Python 3.11 is the current stable release with full ecosystem support
- Python 3.14 (also installed on the machine) is too new — PyTorch and Transformers do not yet publish official wheels for it
- 3.11 is the safest, most compatible choice for this ML stack

**Commands run:**
```powershell
# Move conda environments to D: drive (C: had only 2.09GB free)
conda config --add envs_dirs D:\conda-envs
conda config --add pkgs_dirs D:\conda-pkgs
mkdir D:\conda-envs
mkdir D:\conda-pkgs

# Create environment
conda create -n nishkarsh python=3.11 -y
conda activate nishkarsh

# Install libraries
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate

# Register Jupyter kernel
python -m ipykernel install --user --name=nishkarsh --display-name "Nishkarsh AI"
```

**GPU verification:**
```powershell
nvidia-smi
# Output: NVIDIA GeForce RTX 3050 Laptop GPU, Driver 555.99, CUDA 12.5

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Output: True | NVIDIA GeForce RTX 3050 Laptop GPU
```

**PyTorch version installed:** 2.5.1+cu121
**CUDA note:** PyTorch cu121 build is compatible with CUDA 12.5 driver (CUDA is backward compatible — a 12.5 driver can run 12.1 builds)

**Hurdle encountered:**
First install attempt failed with `[Errno 28] No space left on device` — C: drive had only 2.09GB free and PyTorch requires ~3GB.

**Resolution:** Moved conda environments and package cache to D: (244GB free). Deleted the broken environment, recreated it on D:, and reinstalled all packages successfully.

**Verification of D: placement:**
```powershell
conda info --envs
# Output: nishkarsh  *  D:\conda-envs\nishkarsh  ← confirmed on D:
```

---

### Step 3: Dataset Download

**What we did:**
Downloaded the CFPB Consumer Complaint Database from Kaggle using the Kaggle CLI.

**Dataset details:**
- Source: Kaggle — `namigabbasov/consumer-complaint-dataset`
- Size: 669MB compressed, ~2.5GB uncompressed
- Rows: 2,023,066 complaints
- Date range: 2011–2024
- Columns: 11 total (narrative, Product, Date received, Sub-product, Issue, Sub-issue, Company, State, Timely response?, product_5, Unnamed: 0)

**Commands run:**
```powershell
pip install kaggle

# Set up Kaggle credentials
mkdir C:\Users\dixit\.kaggle
# Downloaded kaggle.json from kaggle.com/settings → API Tokens → Create Legacy API Key
Move-Item C:\Users\dixit\Downloads\kaggle.json C:\Users\dixit\.kaggle\kaggle.json

# Download dataset
kaggle datasets download -d namigabbasov/consumer-complaint-dataset -p data\raw --unzip
```

**Hurdle encountered:**
Kaggle's newer "API Tokens" section generates a token shown as a link, not a downloadable JSON. The "Legacy API Credentials" section is what downloads the actual `kaggle.json` file.

**Resolution:** Used "Create Legacy API Key" button (bottom of kaggle.com/settings/api page) which downloads `kaggle.json` directly to Downloads folder.

**Why CFPB data for an India project:**
See Section 8 for the full rationale. Short answer: no large public Indian complaint dataset with raw text exists. CFPB provides 2M+ real complaint narratives for NLP pretraining; we adapt to Indian categories through mapping and synthetic data augmentation.

---

### Step 4: Jupyter Notebook & EDA

**What we did:**
Launched Jupyter Notebook, created `01_data_exploration.ipynb`, and ran 4 cells of exploratory data analysis.

**What Jupyter is:**
An interactive coding environment that runs in the browser. Code is written and executed in "cells" — small chunks that run independently and show output immediately below. Ideal for data exploration because you can inspect results at each step without re-running the entire script.

**How to launch:**
```powershell
jupyter notebook
```
This starts a local server and opens a browser tab. Navigate to `notebooks/` → New → Nishkarsh AI kernel.

**Cell 1 — Load data:**
```python
import pandas as pd
df = pd.read_csv('../data/raw/complaints.csv', low_memory=False)
print(df.shape)
df.head(3)
```
**Output:** `(2023066, 11)` — 2,023,066 rows, 11 columns. Took ~45 seconds to load (2.5GB file).

**Cell 2 — Columns and missing values:**
```python
print(df.columns.tolist())
print(df.isnull().sum())
```
**Output:**
- Column names: `['Unnamed: 0', 'product_5', 'narrative', 'Product', 'Date received', 'Sub-product', 'Issue', 'Sub-issue', 'Company', 'State', 'Timely response?']`
- `narrative`: 0 missing ✅ (all rows have complaint text)
- `Product`: 0 missing ✅ (all rows have category label)
- `Sub-issue`: 230,559 missing (irrelevant — we don't use this)
- `Sub-product`: 52,206 missing (irrelevant)

**Key finding:** Zero missing values in our two critical columns. No filtering needed.

**Hurdle encountered:**
`df.isnull.sum()` threw `AttributeError: 'function' object has no attribute 'sum'`. The parentheses were missing — `isnull` is a method and must be called as `isnull()` before chaining `.sum()`.

**Resolution:** Corrected to `df.isnull().sum()`.

**Cell 3 — Category distribution:**
```python
print(df['Product'].value_counts())
```
**Output:** 20 unique CFPB product categories. Heavily imbalanced — "Credit reporting..." has 807,291 rows while "Virtual currency" has only 16.

**Cell 4 — Sample complaint text:**
```python
for i, row in df[['narrative', 'Product']].sample(5, random_state=42).iterrows():
    print(f"PRODUCT: {row['Product']}")
    print(f"TEXT: {row['narrative'][:300]}")
```
**Key observations from samples:**
- `XXXX` appears throughout — CFPB anonymises personal information (names, dates, account numbers)
- Text is informal, run-on, and unpunctuated — typical of real complaint writing
- Contains dollar amounts like `{$1200.00}` and bank names
- DistilBERT is well-suited to handle this kind of noisy, informal text

---

### Step 5: Category Mapping

**What we did:**
Created `src/category_mapping.py` and applied a mapping to convert 20 CFPB product labels into 8 of the 15 RBI Ombudsman categories (plus "Others" as catch-all).

**Why this is needed:**
The CFPB uses US-specific product names. The RBI Ombudsman uses completely different Indian regulatory category names. Without this mapping, we'd train a model that predicts "Checking or savings account" instead of "Deposit Accounts" — categories that don't exist in the Indian regulatory framework.

**Mapping logic applied in notebook (inline):**
```python
CFPB_TO_RBI = {
    "Credit card": "Credit Cards",
    "Credit card or prepaid card": "Credit Cards",
    "Checking or savings account": "Deposit Accounts",
    "Bank account or service": "Deposit Accounts",
    "Mortgage": "Loans and Advances",
    "Student loan": "Loans and Advances",
    "Vehicle loan or lease": "Loans and Advances",
    "Consumer Loan": "Loans and Advances",
    "Payday loan": "Loans and Advances",
    "Payday loan, title loan, or personal loan": "Loans and Advances",
    "Payday loan, title loan, personal loan, or advance loan": "Loans and Advances",
    "Debt collection": "Recovery Agents",
    "Debt or credit management": "Recovery Agents",
    "Money transfer, virtual currency, or money service": "Remittance / Money Transfer",
    "Money transfers": "Remittance / Money Transfer",
    "Prepaid card": "ATM / Debit Cards",
    "Virtual currency": "Electronic Banking / Mobile",
    "Other financial service": "Others",
    "Credit reporting": "Others",
    "Credit reporting, credit repair services, or other personal consumer reports": "Others",
    "Credit reporting or other personal consumer reports": "Others",
}

df_clean['rbi_category'] = df_clean['product'].map(CFPB_TO_RBI)
```

**Unmapped check:**
```python
print(df_clean[df_clean['rbi_category'].isna()]['product'].value_counts())
# Output: Series([], ...) — zero unmapped rows ✅
```

**Result — RBI category distribution:**
| RBI Category | Count |
|---|---|
| Others | 1,205,567 |
| Recovery Agents | 267,746 |
| Loans and Advances | 227,695 |
| Credit Cards | 159,041 |
| Deposit Accounts | 115,332 |
| Remittance / Money Transfer | 43,000 |
| ATM / Debit Cards | 4,669 |
| Electronic Banking / Mobile | 16 |

**Note on "Others" being large:** Credit reporting (807k + 366k rows) doesn't have a direct RBI equivalent, so it correctly maps to "Others". This is expected and won't harm training — the model will learn that credit-reporting-style complaints belong to the catch-all category.

**Note on class imbalance:** This is a known issue we'll address during training (Day 1 Steps 6–10) using weighted loss functions or oversampling techniques.

---

### Step 6: Clean Dataset Save

**What we did:**
Saved the final training-ready dataset with only the two columns the model needs: `text` and `rbi_category`.

```python
df_clean[['text', 'rbi_category']].to_csv('../data/processed/complaints_mapped.csv', index=False)
```

**Output file:** `data/processed/complaints_mapped.csv`
**Rows:** 2,023,066
**Columns:** `text` (complaint narrative), `rbi_category` (RBI Ombudsman category)

This file is the direct input to the DistilBERT tokenizer in Steps 6–10.

---

## 6. Hurdles & Resolutions

### Hurdle 1 — PowerShell mkdir syntax
**What happened:** Running `mkdir data notebooks models src` in PowerShell threw `ParameterBindingException: A positional parameter cannot be found that accepts argument 'notebooks'`. PowerShell's mkdir does not accept multiple space-separated arguments.

**Resolution:** Used PowerShell comma-separated syntax: `mkdir data, notebooks, models, src, data\raw, data\processed`

**Learning:** PowerShell and bash have different syntax for the same operations. Throughout this project, bash commands from tutorials must be adapted for PowerShell.

---

### Hurdle 2 — Accidentally pasting terminal prompt
**What happened:** The text `PS C:\Users\dixit\nishkarsh-ai>` was accidentally copied and pasted into the terminal. PowerShell attempted to run it as a process name and threw `NoProcessFoundForGivenName`.

**Resolution:** The prompt text is display-only — it shows your current location. Only the command text that comes after the `>` symbol should be typed or pasted.

---

### Hurdle 3 — Python version selection
**What happened:** The machine had Python 3.11 and 3.14 installed. Original instructions said Python 3.10, causing confusion.

**Resolution:** Chose Python 3.11. Reasoning: 3.14 is too new (no official PyTorch/Transformers wheels), 3.10 is unnecessary when 3.11 is available and fully supported. 3.11 has the widest ML library compatibility as of June 2026.

---

### Hurdle 4 — Disk space error during PyTorch install
**What happened:** `pip install torch` failed mid-install with `[Errno 28] No space left on device`. C: drive had only 2.09GB free; PyTorch requires ~3GB.

**Resolution:**
1. Ran `Get-PSDrive C` and `Get-PSDrive D` to check both drives
2. D: had 244GB free
3. Moved conda environments and package cache to D: permanently:
   ```powershell
   conda config --add envs_dirs D:\conda-envs
   conda config --add pkgs_dirs D:\conda-pkgs
   ```
4. Deleted the broken environment: `conda env remove -n nishkarsh`
5. Recreated on D: and reinstalled all packages successfully

---

### Hurdle 5 — Kaggle token as link, not JSON
**What happened:** Kaggle's new "API Tokens" section generates a token displayed as a link, not a downloadable file. Running `kaggle datasets download` threw an authentication error.

**Resolution:** Used the "Legacy API Credentials" section on kaggle.com/settings/api → "Create Legacy API Key" button → this downloads `kaggle.json` directly. Placed at `C:\Users\dixit\.kaggle\kaggle.json`.

---

### Hurdle 6 — isnull() syntax error
**What happened:** `df.isnull.sum()` threw `AttributeError: 'function' object has no attribute 'sum'`. Written without the calling parentheses.

**Resolution:** `df.isnull().sum()` — `isnull` is a method and must be called with `()` before chaining `.sum()`.

---

### Hurdle 7 — Trying to open 2.5GB CSV in Excel
**What happened:** Windows associated the `.csv` file with Excel and attempted to open it. Excel crashed / couldn't handle 2M rows.

**Resolution:** The file is not meant to be opened in Excel. pandas loads it into memory efficiently using chunked reading internally. Always load large CSVs through Python/pandas, never through spreadsheet software.

---

## 7. Tool Decisions — Why Each Was Chosen

### conda
**Category:** Environment management
**Why:** Creates isolated Python environments so each project's libraries don't conflict with each other or with the system Python. Without conda, installing PyTorch for this project could break a different project that needs an older version. Industry standard for ML/data science projects.

### Python 3.11
**Category:** Language runtime
**Why:** Stable, fully supported by all ML libraries (PyTorch, Transformers, scikit-learn, FastAPI). 3.14 is too new — most ML libraries don't yet publish wheels for it. 3.10 is slightly older than necessary. 3.11 is the sweet spot.

### pandas
**Category:** Data manipulation
**Why:** The de facto standard for loading, filtering, transforming, and saving tabular data in Python. Handles 2M+ row CSVs efficiently using internal chunking. Essential for the entire data pipeline stage.

### PyTorch (torch)
**Category:** Deep learning framework
**Why:** Chosen over TensorFlow because the Hugging Face Transformers library is PyTorch-first — better documentation, more examples, and wider community support. DistilBERT's official implementation is in PyTorch.

### Hugging Face Transformers
**Category:** Pre-trained models
**Why:** Provides DistilBERT and a clean fine-tuning API. DistilBERT specifically was chosen over full BERT because it is 40% smaller, runs 60% faster, and retains 97% of BERT's accuracy — critical for a machine with only 4GB GPU VRAM.

### Hugging Face datasets
**Category:** Data loading
**Why:** Memory-mapped dataset loading — it doesn't load the entire CSV into RAM at once. Has built-in integration with the Transformers tokenizer. Efficient for large datasets on consumer hardware.

### accelerate
**Category:** GPU training utility
**Why:** Handles moving tensors to the GPU automatically without boilerplate CUDA code. Makes multi-device training setup simple. Especially useful for the RTX 3050 where manual device management adds unnecessary complexity.

### scikit-learn
**Category:** Classical ML / evaluation
**Why:** Will be used for evaluation metrics (classification_report, confusion_matrix, accuracy_score) and for any non-deep-learning baseline comparisons. Not used for the main model but essential for measuring it.

### Jupyter Notebook
**Category:** Interactive development environment
**Why:** Allows cell-by-cell code execution with immediate output. Essential for EDA — you can inspect the data at every step, catch issues early, and iterate quickly without re-running the entire script. All exploration and prototyping happens in notebooks; production code is moved to `src/`.

### matplotlib / seaborn
**Category:** Visualisation
**Why:** Used in EDA notebooks to plot category distributions and understand class imbalance before training. seaborn provides cleaner statistical plots with less code.

### VS Code
**Category:** IDE
**Why:** Native Jupyter notebook support, Python IntelliSense, integrated terminal, file explorer, and Git integration in one application. The most widely used Python IDE.

---

## 8. Dataset Rationale — Why US Data for an India Project

This is the most important conceptual question about the data strategy.

### The gap in Indian data
The RBI Ombudsman data available on data.gov.in and Dataful.in is **aggregated and structured** — it provides annual counts like "X complaints in ATM category in FY2023." It contains **no actual complaint text**. An NLP classification model cannot be trained on aggregate counts.

To train a text classifier, you need: raw complaint text + category labels. No large, publicly available Indian dataset with these two things exists as of June 2026.

### What CFPB provides
The CFPB Consumer Complaint Database provides:
- 2,023,066 real complaint narratives (actual text people wrote)
- Pre-labelled product categories
- Data spanning 2011–2024

It is the largest publicly available labelled complaint dataset in the world.

### Why the language transfers
A complaint about ATM fraud reads similarly whether the person is in Mumbai or Miami. The **language patterns of complaint writing** (frustrated tone, specific amounts mentioned, description of sequence of events, references to customer service failures) transfer across geographies. What changes is the regulatory category name — which we handle through our mapping file.

### The adaptation pipeline
```
CFPB narratives (2M rows, English)
        ↓
DistilBERT learns complaint language patterns
        ↓
Category mapping: CFPB labels → RBI categories (category_mapping.py)
        ↓
Synthetic Indian complaints via Gemini API (Day 2)
        ↓
Final model understands Indian banking complaint language and RBI categories
```

This is standard NLP practice called **transfer learning with domain adaptation**. The alternative — manually labelling thousands of Indian complaints from scratch — would take months and is impractical for a 4-day build.

---

## 9. Data Findings from EDA

### Dataset shape
- **2,023,066 rows × 11 columns**
- All rows have complaint narrative text (zero nulls in `narrative` column)
- All rows have product label (zero nulls in `Product` column)

### CFPB categories (20 total, before mapping)
Heavily imbalanced. Top 3 categories account for 72% of all data:
1. Credit reporting, credit repair services... — 807,291 (40%)
2. Credit reporting or other personal consumer reports — 366,397 (18%)
3. Debt collection — 266,842 (13%)

### RBI categories (after mapping)
8 active RBI categories + Others:
- Others (credit reporting mapped here): 1,205,567
- Recovery Agents: 267,746
- Loans and Advances: 227,695
- Credit Cards: 159,041
- Deposit Accounts: 115,332
- Remittance / Money Transfer: 43,000
- ATM / Debit Cards: 4,669
- Electronic Banking / Mobile: 16

### Text quality observations
- CFPB anonymises PII as `XXXX` — model will learn to treat this as a neutral token
- Text is informal and unstructured — representative of real user complaints
- Dollar amounts appear as `{$1200.00}` — model can extract amount-related features
- Average complaint length: several hundred words

### Class imbalance plan
During training (Steps 6–10), we will handle class imbalance using:
- Weighted cross-entropy loss (higher weight to underrepresented classes)
- Stratified train/validation split
- Potential oversampling of ATM/Debit Cards and Electronic Banking/Mobile

---

## 10. Current File State

```
nishkarsh-ai/
├── data/
│   ├── raw/
│   │   └── complaints.csv              # 2,023,066 rows × 11 cols, 2.5GB
│   └── processed/
│       ├── complaints_clean.csv        # 2,023,066 rows × 2 cols (text, product)
│       └── complaints_mapped.csv       # 2,023,066 rows × 2 cols (text, rbi_category) ← TRAINING DATA
│
├── notebooks/
│   └── 01_data_exploration.ipynb      # EDA complete: load, explore, map, save
│
├── models/                            # Empty — model checkpoints go here (Steps 6–10)
│
└── src/
    └── category_mapping.py            # CFPB → RBI mapping dictionary + helper function
```

### Next steps (Day 1, Steps 6–10)
1. **Step 6** — Encode RBI category labels as integers (label encoding)
2. **Step 7** — Tokenise text using DistilBERT tokenizer (truncate to 512 tokens)
3. **Step 8** — Sample training subset (50k–100k rows — full 2M is too large for 4GB VRAM in one session)
4. **Step 9** — Build PyTorch Dataset class and DataLoader
5. **Step 10** — Load DistilBERT with classification head, train 3–5 epochs, evaluate

---

*Document last updated: 20 June 2026 | Day 1 of 4 complete*