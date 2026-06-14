# Smart MCQ Solver — DL & GenAI Project [BSDA2001P]

> **Course** : Deep Learning & Generative AI — Indian Institute of Technology Madras (IITM BS)
> **Term** : T2 2026

---

## Student Details

| Field                     | Value                                                |
| ------------------------- | ---------------------------------------------------- |
| **Name**            | `Risshab Srinivas Ramesh`                          |
| **Roll No**         | `23f3002157`                                       |
| **Kaggle Username** | risshabsrinivas                                      |
| **W&B Profile**     | 23f3002157                                           |
| **GitHub**          | https://github.com/23f3002157/dl-genai-project-26-t2 |

---

## Quick Links

| Resource           | Link                                                                       |
| ------------------ | -------------------------------------------------------------------------- |
| Kaggle Competition | https://www.kaggle.com/competitions/smart-mcq-solver-challenge/overview    |
| Kaggle Notebook    | https://www.kaggle.com/code/risshabsrinivas/dl-23f3002157-notebook-t22026/ |
| W&B Project        | https://wandb.ai/23F3002157-dl-t22026/23f3002157-t22026/overview           |
| GitHub Repo        | https://github.com/23f3002157/dl-genai-project-26-t2                       |

---

## Problem Statement

Build an ML system that solves multiple-choice questions (MCQ) with 5 options (A–E) and predicts the  **top 3 most likely correct answers in ranked order** .

Each question contains:

* A prompt/question
* Five answer choices: A, B, C, D, E

The objective is to rank the top 3 most probable answers correctly.

 **Evaluation Metric** : MAP@3 (Mean Average Precision at 3) — the correct answer ranked higher = better score.

```
Example:
Correct answer: A

A B C  →  highest score  (correct answer at rank 1)
B A C  →  lower score    (correct answer at rank 2)
C D A  →  lowest score   (correct answer at rank 3)
```

---

## Repository Structure

```
dl-genai-project-26-t2/
│
├── notebooks/
│   ├── dl-23f3002157-notebook-t22026.ipynb
│  
│
├── src/
│   ├── utils.py  
│   ├── preprocess.py   
│   ├── train.py  
│   ├── inference.py  
│   ├── baseline_dummy.py 
│   └── upload_to_kaggle.py  
│
├── configs/
│   └── config.yaml   
│
├── data/
│   ├── raw/  
│   └── processed/  
│
├── models/   
├── reports/  
├── logs/     
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Models

| # | Model                         | Type       | Milestone | MAP@3 | W&B Run                                                           |
| - | ----------------------------- | ---------- | --------- | ----- | ----------------------------------------------------------------- |
| 1 | Dummy Baseline (A B C always) | Rule-based | M0        | TBD   | [run-0](https://claude.ai/chat/280aa7ba-cc38-43c7-b50f-83bb7a634236) |
|   |                               |            |           |       |                                                                   |

> All runs tracked and compared on W&B under project `23f3002157-t22026`.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up credentials

Create a `.env` file in the project root:

```
WANDB_API_KEY=your_wandb_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Set up Kaggle credentials locally:

```bash
mkdir -p ~/.kaggle
echo '{"username":"<YOUR_KAGGLE_USERNAME>","key":"<YOUR_KAGGLE_KEY>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Log in to W&B:

```bash
wandb login
```

### 5. Download the dataset

```bash
kaggle competitions download -c <COMPETITION_SLUG> -p data/raw/
unzip data/raw/<COMPETITION_SLUG>.zip -d data/raw/
```

---

## 📅 Milestones

| Milestone                               | Branch          | Deadline     | Description                                               | Status |
| --------------------------------------- | --------------- | ------------ | --------------------------------------------------------- | ------ |
| **M0**— Setup & Dummy Submission | `milestone-0` | Jun 18, 2026 | Env setup, dummy A B C baseline, first Kaggle submission  | ✅     |
| **M1**— EDA & Baseline           | `milestone-1` | Jun 24, 2026 | Data exploration, class distribution, rule-based baseline | ⬜     |
| **M2**— Classical ML             | `milestone-2` | Jul 1, 2026  | TF-IDF + Logistic Regression / XGBoost, W&B logging       | ⬜     |
| **M3**— CNN / NN                 | `milestone-3` | Jul 8, 2026  | Custom CNN text classifier, PyTorch training loop         | ⬜     |
| **M4**— Sequential Models        | `milestone-4` | Jul 15, 2026 | CRNN (CNN + LSTM/GRU), compare with CNN                   | ⬜     |
| **M5**— Transformer Fine-tune    | `milestone-5` | Jul 21, 2026 | BERT / DeBERTa fine-tuning via Hugging Face               | ⬜     |
| **Final**— Submission            | `main`        | Jul 26, 2026 | Final Kaggle submission, report, optional deployment      | ⬜     |

---

## Submission Format

```
ID,Prediction
1,A B C
2,C A D
3,B D A
```

Each row: question ID + top 3 predicted option labels separated by spaces.

---

## Reports

| Milestone | Report                             |
| --------- | ---------------------------------- |
| M0        | `reports/milestone-0-report.pdf` |
| M1        | `reports/milestone-1-report.pdf` |
| Final     | `reports/final-report.pdf`       |
