# 🧠 Enhancing Public Complaint Efficiency through Government Institution Classification and Named Entity Recognition (NER)  
### 🏆 GEMASTIK XVII 2024 — Data Mining Division (Top 20 Finalist)  
### 👥 Team Kotak Riset SC | Universitas Gadjah Mada  

---

## 📘 Overview

This repository contains the complete implementation and documentation for the research project **“Enhancing Public Complaint Efficiency through Government Institution Classification and Named Entity Recognition (NER)”**, developed by **Team Kotak Riset SC** from **Universitas Gadjah Mada (UGM)**.

This work was submitted to **GEMASTIK XVII 2024 – Data Mining Division**, a **national-level competition officially held by the Ministry of Education, Culture, Research, and Technology of the Republic of Indonesia (KEMENDIKBUDRISTEK / KEMENRISTEKDIKTI)**.  
The project achieved **Top 20 National Finalist** status, ranking among the best out of more than **200 participating teams** from across Indonesia.

The research aims to revolutionize the **SP4N-LAPOR! (National Public Complaint System)** — Indonesia’s primary e-government complaint management platform — by leveraging **Natural Language Processing (NLP)** to automatically classify, extract, and route citizen reports to the appropriate government institutions.

---

## 🎯 Problem Background and Motivation

SP4N-LAPOR! serves as Indonesia’s centralized digital platform for citizens to submit complaints, criticisms, or suggestions regarding public services.  
However, despite its importance, **the routing of reports is still largely manual**, performed by human officers who must read each submission and decide which ministry, agency, or office is responsible.  

This manual process causes several systemic issues:

- ⏱️ **Slow response times:** Manual classification can take **up to 3–5 days** per complaint.  
- ⚖️ **Inconsistency:** Similar cases may be routed to different institutions by different officers.  
- 🧍‍♀️ **High workload:** Thousands of reports arrive daily, creating bottlenecks.  
- 📊 **Data underutilization:** Valuable patterns in complaint data remain unexplored.

To address this, our solution applies **machine learning–based text classification** to automate the routing step and **Named Entity Recognition (NER)** to extract structured metadata (e.g., institution names, regions, dates, event types).  
Together, these methods enable faster, consistent, and scalable complaint management.

---

## 🏅 Achievements

> 🥇 **Top 20 Finalist** — *Pagelaran Mahasiswa Nasional Bidang Teknologi Informasi dan Komunikasi (GEMASTIK XVII 2024)*  
> Officially recognized by **KEMENDIKBUDRISTEK / KEMENRISTEKDIKTI**  
> Selected among **200+ national university teams** in the **Data Mining Division**

---

## 📈 Research Objectives

1. **Automate Institution Classification**  
   Classify public complaints into the correct government institution category.

2. **Develop a Named Entity Recognition (NER) System**  
   Identify entities (locations, institutions, event types, and social issues) mentioned in reports.

3. **Reduce Manual Workload**  
   Minimize human involvement in the triage and routing process.

4. **Build a Scalable NLP Framework**  
   Enable future integration with real-time SP4N-LAPOR! APIs and dashboards.

---

## 🧩 Dataset Description

### 1️⃣ Public Complaint Dataset
- **Source:** Official SP4N-LAPOR! portal (scraped via automated crawler)  
- **Collection Period:** May 2020 – March 2024  
- **Total Records:** 10,812 unique complaints  
- **Attributes:**
  | Column | Description |
  |:--|:--|
  | `judul` | Report title or summary |
  | `isi` | Full complaint text |
  | `kategori` | Government institution (target class) |
  | `tanggal` | Submission date |
  | `provinsi` | Origin province of the complaint |
  | `status` | Handling status (open/closed) |

- **Target Classes (7 main institutions):**
  1. National Police of Indonesia (POLRI)  
  2. Ministry of Education, Culture, Research, and Technology  
  3. Ministry of Social Affairs  
  4. PLN (State Electricity Company)  
  5. Directorate General of Land Transportation  
  6. Ministry of Communication and Information (KOMINFO)  
  7. Directorate General of Health Manpower  

---

### 2️⃣ NER Dataset
- **Source:** [id_nergrit_corpus (2024)](https://huggingface.co/datasets/indonlp/id_nergrit_corpus) by GRIT Indonesia  
- **Language:** Indonesian (Bahasa Indonesia)  
- **Entity Labels:** `PER`, `LOC`, `ORG`, `DAT`, `EVT`, `NUM`  
- **Custom Label Added:** `LAP` (for social/public complaint keywords)  
- **Corpus Size:** 11,643 sentences, 92,000+ tokens  
- **Annotation Format:** BIO tagging scheme  

---

## ⚙️ Data Preprocessing

Text normalization and cleaning were critical to prepare noisy, informal complaint text.

| Step | Description |
|:--|:--|
| Lowercasing | Convert all text to lowercase |
| Unicode Normalization | Normalize Indonesian diacritics |
| Stopword Removal | Use Indonesian stopword list (Bahasa-ID) |
| Lemmatization/Stemming | Applied Sastrawi stemmer |
| Emoji & URL Removal | Using regex filters |
| Tokenization | WordPiece tokenizer (for Transformer models) |
| Padding | Max length = 128 tokens |
| Train-Test Split | 80% train / 20% test |

Additionally, the **`LAP` entity** was introduced for social-issue tokens like *bantuan, subsidi, penipuan, jalan rusak, listrik padam*, etc.

---

## 🧠 Model Architecture

### 1️⃣ Institution Classification
We evaluated four deep learning architectures:

| Model | Description | Strength |
|:--|:--|:--|
| **IndoBERTweet** | Transformer pretrained on 26M Indonesian tweets | Excels on informal, noisy text |
| IndoBERT Base | Transformer pretrained on Wikipedia + news corpus | Performs well on formal text |
| LSTM | Sequential deep RNN | Lightweight, interpretable baseline |
| CNN-LSTM | Hybrid conv-recurrent model | Captures local + sequential features |

**Training Configuration:**
```python
epochs = 5
batch_size = 16
optimizer = AdamW(lr=2e-5)
scheduler = linear_decay
loss_fn = CrossEntropyLoss()
eval_metric = F1_macro
```

**Results:**
| Model | Accuracy | F1 Score | Training Time |
|:------|:---------:|:---------:|:--------------:|
| **IndoBERTweet** | **0.881** | **0.881** | 2450 s |
| IndoBERT | 0.752 | 0.754 | 2415 s |
| LSTM | 0.803 | 0.803 | 1684 s |
| CNN-LSTM | 0.791 | 0.791 | 724 s |

**Analysis:**  
- IndoBERTweet achieved the best overall performance due to its familiarity with colloquial Indonesian.  
- LSTM models were faster but struggled with semantic ambiguity and long dependencies.  
- Transformer-based models generalized well, with fewer misclassifications in overlapping categories.

---

### 2️⃣ Named Entity Recognition (NER)
For entity extraction, we fine-tuned **IndoBERT Uncased Base** on the GRIT dataset with our additional `LAP` entity.

| Hyperparameter | Value |
|:--|:--|
| Epochs | 6 |
| Batch Size | 8 |
| Learning Rate | 3e-5 |
| Optimizer | AdamW |
| Dropout | 0.1 |
| Scheduler | Cosine decay with warmup |

**Performance:**
| Metric | Score |
|:--|:--|
| Accuracy | 0.9494 |
| F1 Score | 0.9163 |
| Precision | 0.9218 |
| Recall | 0.9111 |

**Observation:**  
The model performed exceptionally on `LOC`, `ORG`, and `DAT` tags but showed moderate confusion between `PER` and `LAP` entities when the report mentioned individual names in social contexts.

---

## 🔬 Experimental Setup

| Parameter | Configuration |
|:--|:--|
| Framework | PyTorch + HuggingFace Transformers |
| Hardware | NVIDIA Tesla T4 GPU (Colab Pro) |
| Environment | Python 3.10, CUDA 11.8 |
| Logging | WandB integration for training visualization |
| Cross-validation | 5-fold Stratified |
| Tokenizer | IndoBERTweetTokenizer & IndoBERTTokenizer |

---

## 🧮 Evaluation and Analysis

### Feature Importance
We used **SHAP (SHapley Additive exPlanations)** to interpret IndoBERTweet’s decision process.  
Words like **“sekolah”, “pungli”, “bantuan”, “listrik”, “kementerian”, “jalan”** were identified as high-impact tokens correlating with specific government institutions.

### Error Analysis
- 7% of misclassifications occurred between **Social Affairs vs. Health Workforce**, where textual overlap was high.  
- Complaints containing multiple institutional references were often assigned to the first dominant context.

### Ablation Studies
| Configuration | F1 Score |
|:--|:--|
| Base IndoBERT (no pretraining) | 0.74 |
| + Domain-Specific Fine-Tuning | 0.79 |
| + IndoBERTweet Pretraining | **0.88** |
| + Custom Token Cleaning (LAP Entity) | **+4% improvement in NER F1** |

---

## 🚀 Integration Pipeline

```text
Citizen Complaint (Raw Text)
     ↓
Preprocessing & Tokenization
     ↓
Institution Classification (IndoBERTweet)
     ↓
Named Entity Recognition (IndoBERT)
     ↓
Entity Extraction (ORG, LOC, DAT, LAP)
     ↓
Automatic Routing to Responsible Institution
     ↓
Dashboard Visualization (Service Load, Topic Trends)
```

The final system produces structured JSON output:

```json
{
  "institution": "Kementerian Sosial",
  "entities": {
    "LOC": "Jakarta Selatan",
    "DAT": "12 Maret 2024",
    "LAP": "bantuan sosial tidak diterima"
  },
  "confidence": 0.8871
}
```

---

## 📊 Results Summary

| Task | Model | Metric | Score |
|:--|:--|:--|:--|
| Institution Classification | IndoBERTweet | F1 | **0.881** |
| NER | IndoBERT Base | F1 | **0.9163** |
| End-to-End Routing Accuracy | Combined | **86.5%** |

---

## 🌍 Impact and Discussion
- Automated routing reduced average classification time from **3 days → 5 seconds**.  
- Predicted **institution category** and **key entities** allow early prioritization of critical complaints (e.g., safety, fraud).  
- Offers a **transparent and auditable AI model** through SHAP visualizations.  
- Contributes toward **Indonesia’s “Smart Bureaucracy 2025” roadmap**, promoting data-driven governance.

---

## ⚙️ Tools & Libraries
```python
pandas, numpy, scikit-learn, seaborn, matplotlib  
torch, transformers, pytorch_lightning  
Sastrawi, tqdm, wandb, shap, json, regex
```

---

## 📘 Limitations and Future Work
- Current dataset limited to **7 major institutions**; future iterations will expand to **>50 government agencies**.  
- Multilingual reports (mixed Indonesian-English) require domain adaptation.  
- Integration with **real-time SP4N-LAPOR! API** and **dashboard UI** planned.  
- Ongoing research into **multi-task learning (NER + classification)** to reduce model redundancy.

---

## 🧾 References
1. Kominfo Indonesia (2024). *SP4N-LAPOR! Public Complaint System.*  
2. GRIT Indonesia (2024). *id_nergrit_corpus Dataset.* HuggingFace.  
3. Koto et al. (2020). *IndoBERT: A Pre-trained Language Model for Indonesian NLP.* COLING.  
4. Koto et al. (2021). *IndoBERTweet: A Domain-Specific BERT Model for Indonesian Twitter.* EMNLP.  
5. Ministry of Education, Culture, Research, and Technology (2024). *GEMASTIK XVII Competition Guidelines.*  

---

## 👥 Authors and Contributions
| Name | Role | Key Contributions |
|:--|:--|:--|
| **Krisna Bayu Dharma Putra** | Artificial Intelligence Engineer | Model architecture design, training pipeline, Classification modeling, dataset scraper, performance optimization, NER fine-tuning, paper writer |
| **Nailfaaz** | Artificial Intelligence Engineer | Dataset curation, NER fine-tuning, literature study, paper writer |
| **Rabbani Nur Kumoro** | Quality Assurance | Quality Assurance, paper writer |
| **Yunita Sari** | Research Advisor | Supervision, validation, and academic guidance |

📧 [linkedin.com/in/dharma-putra1305](https://linkedin.com/in/dharma-putra1305)  
🌐 [github.com/kbdp1305](https://github.com/kbdp1305)
