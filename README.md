# 🔍 NLP Intelligent Talent Search System
### Multimodal Semantic Candidate Ranking for Senior Network Security Roles

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange?style=flat)](https://trychroma.com)
[![HuggingFace](https://img.shields.io/badge/Embeddings-sentence--transformers-yellow?style=flat&logo=huggingface)](https://sbert.net)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black?style=flat)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smuciton/nlp-intelligent-talent-search/blob/main/NLP_Intelligent_Talent_Search_Senior_Network_Security.ipynb)

---

## Overview

An end-to-end **NLP pipeline** that simulates the semantic ranking layer of a modern
**Applicant Tracking System (ATS)** — without keyword matching.

The system reads a real job posting (*Consulting Systems Engineer*),
ingests a corpus of 20 anonymized synthetic CVs in mixed formats (**PDF + PNG**),
builds a persistent **ChromaDB vector database**, and ranks candidates by semantic
similarity using **Transfer Learning** — finally generating **LLM-powered justifications**
for the top-5 recommendations via a **RAG (Retrieval-Augmented Generation)** pipeline.

> ⚠️ **Disclaimer:** All 20 CVs are fully synthetic and anonymized — no real personal
> data (PII) of any kind. The Fortinet job posting is publicly available on their careers
> portal and is used here for research and demonstration purposes only. This project has
> no affiliation with Fortinet and does not reflect their internal systems or processes.

---

## Why Transfer Learning — Not Training from Scratch

Training a domain-specific language model from scratch would require:
- Billions of tokens of labeled training data
- Weeks of GPU compute time
- Significant infrastructure cost

Instead, this project uses **`sentence-transformers/all-MiniLM-L6-v2`** — a model
pre-trained on 1B+ sentence pairs that already encodes semantic meaning out of the box.
This means a CV describing *"cloud-delivered security"* and *"identity-centric access"*
will rank high against a posting using *SASE* and *ZTNA* — **zero fine-tuning required.**

This is Transfer Learning in practice: take what a large model already knows,
apply it directly to a specialized problem.

---

## Pipeline Architecture

```
MULTIMODAL TALENT SEARCH PIPELINE
═══════════════════════════════════════════════════════════════

 STEP 1-2  DOCUMENT INGESTION
 ┌─────────────────────────────────────────────────────────┐
 │  read_pdf()          read_png()        extract_level()  │
 │  pdfplumber          Tesseract OCR     CV number range  │
 │  16 PDF files        4 PNG files       filters non-CVs  │
 └──────────────────────────┬──────────────────────────────┘
                            │  raw text + metadata
                            ▼
 STEP 3  TEXT PREPROCESSING
 ┌─────────────────────────────────────────────────────────┐
 │  clean_text()                                           │
 │  lowercase · remove separators · filter stopwords       │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 STEP 4  EMBEDDINGS + VECTOR DATABASE
 ┌─────────────────────────────────────────────────────────┐
 │  SentenceTransformer('all-MiniLM-L6-v2')  →  384 dims  │
 │  collection.add()  →  ChromaDB (persistent)             │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 STEP 5-6  SEMANTIC RETRIEVAL
 ┌─────────────────────────────────────────────────────────┐
 │  job_posting  →  retrieve_candidates(query, k=5)        │
 │  collection.query()  ·  cosine distance                 │
 └──────────────────────────┬──────────────────────────────┘
                            │  df_ranking: 20 CVs by score
                            ▼
 STEP 7  LLM-POWERED JUSTIFICATION  (RAG pattern)
 ┌─────────────────────────────────────────────────────────┐
 │  build_prompt(cv_info, job_posting)                     │
 │                                                         │
 │   Option A          Option B         Fallback           │
 │   Claude API        Ollama local     rule-based          │
 │       └─────────────────┴──────────────┘               │
 │                         ▼                               │
 │         generate_response()  →  natural language        │
 └─────────────────────────────────────────────────────────┘
                            │
                            ▼
 OUTPUT: Top-5 candidates · similarity score · justification
═══════════════════════════════════════════════════════════════
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Text Extraction** | `pdfplumber` | PDF direct extraction |
| **OCR** | `pytesseract` + Tesseract 4 | PNG image-to-text |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Semantic vectorization |
| **Vector Database** | `ChromaDB` (persistent) | Candidate indexing & search |
| **LLM Backend A** | Anthropic Claude API | Cloud LLM justifications |
| **LLM Backend B** | Ollama (`qwen2.5:0.5b`) | Local LLM — no data sent externally |
| **Runtime** | Google Colab | Cloud notebook execution |
| **Visualization** | `matplotlib` | Ranking charts & heatmaps |

---

## Repository Structure

```
nlp-intelligent-talent-search/
│
├── README.md
├── NLP_Intelligent_Talent_Search_Senior_Network_Security.ipynb
│
└── synthetic_cvs/
    ├── gemini/          ← CVs generated with Google Gemini
    │   ├── CV01_Junior_NetworkSecurity_Engineer.png
    │   ├── CV02_Junior_Cybersecurity_Analyst.pdf
    │   └── ...
    └── claude/          ← CVs generated with Anthropic Claude
        ├── CV01_Junior_NetworkSecurity_Engineer.png
        ├── CV02_Junior_Cybersecurity_Analyst.pdf
        └── ...
```

---

## Synthetic CV Corpus — Design

| Experience Level | Count | CV IDs | Years |
|---|---|---|---|
| **Junior** | 5 | CV01 – CV05 | 0–2 years |
| **Intermediate** | 10 | CV06 – CV15 | 3–6 years |
| **Senior** | 5 | CV16 – CV20 | 7+ years |

| Profile Type | CVs | Relevance to Posting |
|---|---|---|
| Network Security Engineer / Architect | CV01, CV06, CV09, CV14, CV16 | ⭐⭐⭐⭐⭐ Very High |
| Security Engineer / Cybersecurity | CV02, CV08, CV17, CV20 | ⭐⭐⭐⭐ High |
| Cloud Security / DevSecOps | CV03, CV07, CV10, CV18 | ⭐⭐⭐⭐ High |
| IT Systems Engineer (generalist) | CV11 | ⭐⭐⭐ Medium |
| Software Developer / Architect | CV04, CV12, CV19 | ⭐⭐ Low |
| Data / BI Analyst | CV05, CV13, CV15 | ⭐ Very Low |

> The intentional mix of relevant and unrelated profiles enables validation
> that the system correctly surfaces technical matches and filters out irrelevant ones.

---

## Key Results

- ✅ Network Security / SASE profiles consistently ranked **#1–#5**
- ✅ BI Analysts and Data Engineers ranked **#16–#20** — correctly filtered
- ✅ **Precision@5 ≥ 0.8** against expert-assigned ground truth labels
- ✅ Semantic embeddings captured domain alignment **without exact keyword overlap**
- ⚠️ PNG/OCR-extracted CVs ranked slightly lower due to OCR noise — expected behavior

---

## How to Run

### Option 1 — Google Colab (recommended)

Click the **Open in Colab** badge above, then:

1. Upload the `synthetic_cvs/` folder to your Google Drive
2. Update `BASE_DIR` in **Section 3.1** to your Drive path
3. Run all cells sequentially
4. Choose your LLM backend in **Section 3.7**:
   - `USAR_OLLAMA = True` for local inference (no API key needed)
   - `USAR_API_CLAUDE = True` + `ANTHROPIC_API_KEY` for Claude

### Option 2 — Local

```bash
git clone https://github.com/smuciton/nlp-intelligent-talent-search.git
cd nlp-intelligent-talent-search
pip install sentence-transformers chromadb pdfplumber pytesseract pillow
# Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
jupyter notebook NLP_Intelligent_Talent_Search_Senior_Network_Security.ipynb
```

---

## Privacy & Ethics

All candidate profiles are **100% synthetic** — generated by an LLM for
research and demonstration purposes only. They contain:

- ❌ No real names, photos, ages, or genders
- ❌ No nationalities, addresses, or contact details
- ❌ No real work history or education records

**Why anonymized?** Embedding models can encode correlations between names and
demographic attributes, introducing bias into rankings. Anonymization ensures the
system evaluates only skills, experience, certifications, and domain alignment.

---

## Proposed Improvements

1. **Post-OCR spell correction** — reduce noise from scanned CV images
2. **Section-level chunking** — index CV sections separately for finer-grained retrieval
3. **Domain-specialized embeddings** — fine-tune on cybersecurity job/CV pairs
4. **Human-in-the-loop evaluation** — Spearman correlation vs. independent expert ranking
5. **Larger corpus** — scale to 200+ CVs for statistically reliable evaluation
6. **🖥️ Interactive Web Interface** — The most impactful next step is wrapping this pipeline into a **web application** that allows HR teams and recruiters to use the system without touching any code.

---

## Author

**Sebastian Mucito**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/sebastian-mucito)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/smuciton)

---

## License

*Built with Python · sentence-transformers · ChromaDB · pytesseract · pdfplumber · Ollama · Anthropic Claude*
