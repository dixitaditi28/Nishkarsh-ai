<div align="center">

# 🏛️ Nishkarsh AI
### Intelligent Banking Complaint Triage & Resolution Predictor

[![Status](https://img.shields.io/badge/Status-WIP%20🚧-yellow)](https://github.com)
[![Day](https://img.shields.io/badge/Build%20Phase-Day%201%20of%204-blue)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%20CUDA-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

*Nishkarsh (Sanskrit: निष्कर्ष) — Conclusion / Outcome*

</div>

---

## 📌 What is Nishkarsh AI?

Nishkarsh AI is a complaint intelligence pipeline built for the Indian banking regulatory context. It takes raw, unstructured consumer complaint text as input — including voice input in Hindi and regional languages — and outputs:

| Output | Description |
|--------|-------------|
| 🏷️ **Category** | Classifies into 1 of 15 RBI Ombudsman categories |
| 🔥 **Urgency Score** | 1–10 score based on amount, type, repeat filer |
| ⏱️ **Resolution Time** | Predicted days to resolution |
| ⚖️ **Outcome** | Predicted: in complainant's favour or not |
| 📋 **Officer Brief** | LLM-generated structured summary for reviewing officer |

---

## 🎯 Problem Statement

When a consumer files a complaint with the RBI Ombudsman or on the CMS portal, it lands in a queue with no intelligent routing. Banks manually classify complaints by type, and resolution times vary wildly — ATM fraud complaints sitting next to KYC paperwork disputes.

There is no system that predicts:
- Which category a complaint belongs to
- How urgent it is
- Whether it's likely to be resolved in favour of the consumer

Small-town and first-time complainants write vague, unstructured text — the system fails them first.

---

## 🗺️ Project Roadmap

```
Day 1 ✅  →  Data Pipeline & ML Foundations
Day 2 🔜  →  Regression Heads, Outcome Model & Whisper ASR
Day 3 🔜  →  FastAPI Backend & PostgreSQL
Day 4 🔜  →  Streamlit Dashboard & AWS Deployment
```

---

## 🧠 ML Architecture

```
Raw Text / Voice Input
        │
        ▼
[Whisper ASR] ←── Hindi / Regional Language Audio
        │
        ▼
  Complaint Text
        │
        ├──► [DistilBERT Classifier] ──► RBI Category (15 classes)
        │
        ├──► [Regression Head]       ──► Resolution Time (days)
        │
        ├──► [Binary Classifier]     ──► Outcome (favour / not)
        │
        └──► [Gemini API Summarizer] ──► Structured Officer Brief
```

---

## 📁 Project Structure

```
nishkarsh-ai/
├── data/
│   ├── raw/
│   │   └── complaints.csv          # CFPB dataset (2M rows, ~2.5GB)
│   └── processed/
│       ├── complaints_clean.csv    # Cleaned: text + product columns
│       └── complaints_mapped.csv  # Final: text + rbi_category (training-ready)
│
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA, category mapping, data cleaning
│
├── models/                        # Saved model checkpoints (Day 1 end)
│
└── src/
    └── category_mapping.py        # CFPB → RBI category mapping logic
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Models** | DistilBERT (HuggingFace), scikit-learn, Whisper ASR |
| **LLM** | Gemini API (officer brief summarization) |
| **Backend** | FastAPI, PostgreSQL |
| **Frontend** | Streamlit dashboard |
| **Infra** | Docker, AWS EC2 + S3 |
| **Language** | Python 3.11 |
| **GPU** | NVIDIA RTX 3050 (CUDA 12.5) |

---

## 📊 Dataset

| Dataset | Source | Purpose |
|---------|--------|---------|
| CFPB Consumer Complaint Database | Kaggle | 2M+ labelled complaint narratives for NLP pretraining |
| RBI Ombudsman Data | data.gov.in / Dataful.in | Category structure reference |
| Synthetic Indian Complaints | Gemini API (Day 2) | Domain adaptation for Indian banking context |

**Why US CFPB data for an India-focused project?**
The CFPB dataset provides 2M+ real complaint narratives — the largest publicly available labelled complaint dataset. There is no equivalent Indian dataset with raw complaint text. We use CFPB data for pretraining (the model learns complaint language patterns), then adapt to RBI categories through mapping and synthetic Indian data augmentation. See `DOCUMENTATION.md` for the full rationale.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11 (Anaconda recommended)
- NVIDIA GPU with CUDA 12.x (or CPU, slower)


### Step 1: Clone the repository
```bash
git clone https://github.com/dixitaditi28/nishkarsh-ai.git
cd nishkarsh-ai
```

### Step 2: Create conda environment
```bash
conda create -n nishkarsh python=3.11 -y
conda activate nishkarsh
```

### Step 3: Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
python -m ipykernel install --user --name=nishkarsh --display-name "Nishkarsh AI"
```

### Step 4: Download dataset
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d namigabbasov/consumer-complaint-dataset -p data/raw --unzip
```

### Step 5: Run EDA notebook
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 🎯 Target Metrics

| Metric | Target |
|--------|--------|
| Classification Accuracy | 92%+ across 15 RBI categories |
| Voice Pipeline Speed | < 3 seconds end-to-end |
| Resolution Time MAE | ± 2 days |
| Dashboard KPI | Top 3 categories + avg resolution time by bank |

---

## 📈 Current Progress (Phase 1)

- [x] Project structure created
- [x] Conda environment configured (D: drive, Python 3.11)
- [x] PyTorch 2.5.1 with CUDA 12.1 installed — RTX 3050 verified
- [x] CFPB dataset downloaded (2,023,066 rows)
- [x] EDA completed — all columns explored, nulls checked
- [x] Category mapping: 20 CFPB labels → 8 RBI categories (+ Others)
- [x] Training-ready dataset saved: `data/processed/complaints_mapped.csv`
- [ ] DistilBERT tokenization (Phase 1, Steps 6–10)
- [ ] Model fine-tuning
- [ ] Regression & outcome heads (Phase 2)
- [ ] Whisper ASR integration (Phase 2)
- [ ] FastAPI backend (Phase 3)
- [ ] Streamlit dashboard (Phase 4)
- [ ] AWS deployment (Phase 4)

---

## 👨‍💻 Developer

**Dixit** | 

---

## 📄 Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for a detailed step-by-step log of everything done, every hurdle faced, how each was resolved, and the rationale behind every tool and decision.

---

> **🚧 Work in Progress** — This project is being built over 4 Phases. Check back for updates.