# HireSense — AI-Powered Resume Screening

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)]()
[![Framework: FastAPI | Streamlit](https://img.shields.io/badge/framework-FastAPI%20%7C%20Streamlit-orange)]()
[![Docker Enabled](https://img.shields.io/badge/docker-enabled-lightblue)]()

Summary
-------
HireSense is an AI-driven resume screening system that extracts structured information from resumes and ranks candidates against job descriptions using semantic NLP (SBERT). It produces explainable scores and highlights which features contributed to a candidate's ranking.

Key features
------------
- PDF resume parsing (PyMuPDF + Tesseract fallback)
- Sectioning: Experience, Projects, Education, Skills
- Feature extraction using dictionaries and rules
- Semantic matching with Sentence-BERT embeddings
- Weighted, explainable scoring with per-feature contributions
- Streamlit UI for uploads and review
- FastAPI backend with JWT-based authentication
- Dockerized development and deployment

Repository layout
-----------------
hiresense/
- README.md
- requirements.txt
- config/
  - hiresense_skills_dictionary_v2.json
  - project_triggers.json
  - profile_config.json
  - feature_strength_rules.json
- data/
  - raw_resumes/
  - jds/
  - labeled/
- src/
  - api/        # FastAPI services
  - pipeline/   # Orchestration of phases
  - phases/     # Text extraction, sectioning, feature extraction
  - ui/         # Streamlit app
  - utils/      # Helpers
- tests/
  - unit/
  - integration/
- docs/

Architecture (high level)
-------------------------
1. Text extraction: PyMuPDF -> plaintext (Tesseract OCR fallback)
2. Sectioning: rule-based + ML-assisted detection of sections
3. Feature extraction: NER, dictionaries, heuristics -> JSON profile
4. Semantic matching: SBERT embeddings + cosine similarity against JD
5. Scoring engine: configurable weights -> final score + breakdown
6. UI & API: Streamlit dashboard for recruiters, FastAPI for programmatic access

Quickstart (development)
------------------------
Prerequisites:
- Python 3.10+
- Docker & Docker Compose (optional for DB/services)
- Tesseract OCR (optional, for scanned PDFs)

1. Create and activate a virtual environment (Windows PowerShell)
   powershell
   $ python -m venv .hirevenv
   $ .\.hirevenv\Scripts\Activate.ps1

   (CMD)
   > python -m venv .hirevenv
   > .\.hirevenv\Scripts\activate

2. Install dependencies
   powershell
   $ pip install -r requirements.txt

3. Start supporting services with Docker (optional)
   powershell
   $ docker compose up -d

4. Set environment variables (PowerShell example)
   powershell
   $env:AUTH_API_BASE = "http://127.0.0.1:8000"
   $env:RESUME_API_BASE = "http://127.0.0.1:8001"

5. Run tests
   powershell
   $ pytest -q tests/unit

6. Run services
   - Start FastAPI auth (example)
     powershell
     $ uvicorn src.main:app --reload --port 8000

   - Start resume scoring API
     powershell
     $ uvicorn src.api.main:app --reload --port 8001

   - Start Streamlit UI
     powershell
     $ streamlit run streamlit_app.py

Configuration
-------------
- config/feature_strength_rules.json — per-feature scoring weights
- config/hiresense_skills_dictionary_v2.json — curated skill lists and aliases
- config/profile_config.json — sections and thresholds

Example output (single candidate)
--------------------------------
{
  "candidate_name": "Alex Johnson",
  "match_score": 0.87,
  "feature_contributions": {
    "skills": 0.25,
    "projects": 0.28,
    "experience": 0.15,
    "education": 0.08,
    "open_source": 0.11
  },
  "missing_skills": ["Power BI", "Docker"],
  "remarks": "Strong backend & ML developer; improve cloud exposure."
}

Testing and evaluation
----------------------
- Unit tests: tests/unit/
- Integration tests: tests/integration/
- Metrics: Precision@K, nDCG@K, MRR for ranking; Precision/Recall/F1 for extraction

Development notes
-----------------
- Embeddings: HuggingFace / SentenceTransformers; consider fine-tuning on labeled JD-resume pairs.
- Use FAISS for scalable nearest-neighbor search in production.
- Keep feature weight rules in config for easy experimentation.

Contributing
------------
- Fork, create a feature branch, add tests, and open a PR with a clear description.
- Follow existing code style and add unit tests for new behavior.

License
-------
MIT

Author
------
Jay Kakadiya — 2025
Contact: use repository issues for questions or feature requests
