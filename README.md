# 🛡️ TruthGuard

**TruthGuard** is a modular and extensible fact-checking and trend-detection pipeline designed for verifying claims made on social media platforms like Reddit. It uses NLP and knowledge graph lookups to detect misinformation, generate reports, and send alerts.

---

## 📂 Project Structure
TruthGuard/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── claim.py
│   │   │   ├── verification.py
│   │   │   └── user.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── claim_extractor.py
│   │   │   ├── fact_checker.py
│   │   │   ├── knowledge_graph.py
│   │   │   ├── semantic_matcher.py
│   │   │   ├── trend_detector.py
│   │   │   └── alert_system.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py
│   │   │   └── schemas.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── text_processing.py
│   │       └── helpers.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_claim_extractor.py
│   │   ├── test_fact_checker.py
│   │   └── test_api.py
│   ├── scripts/
│   │   ├── setup_db.py
│   │   └── migrate.py
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── models/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── manifest.json
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.js
│   │   │   ├── ClaimChecker.js
│   │   │   ├── TrendAnalysis.js
│   │   │   ├── ReportViewer.js
│   │   │   └── AlertPanel.js
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── utils/
│   │   │   └── helpers.js
│   │   ├── styles/
│   │   │   └── App.css
│   │   ├── App.js
│   │   ├── index.js
│   │   └── App.css
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── README.md
├── LICENSE
└── .gitignore

---

## 🚀 Features

- 🔍 **Claim Extraction** – Extract factual claims from social posts
- 🧠 **Semantic Matching** – Use sentence embeddings to match claims to verified facts
- 🌐 **Knowledge Graph Lookup** – Validate claims using Wikidata or similar
- ✅ **Fact Check API Integration** – Supports APIs like Google Fact Check Tools
- 📊 **Trend Detection** – Identify trending keywords and topics
- 📝 **Report Generation** – Output user-friendly truth verdicts
- 🔔 **Alert System** – Send notifications for critical misinformation

---

## ⚙️ Getting Started

### 1. Clone the repository
``` bash
git clone https://github.com/arnav.1803/truthguard.git
cd truthguard
```

### 2. Set up a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate  # On Linux/macOS

### 3. Install depenedencies
pip install -r requirements.txt

### 4. Rune the pipeline
python src/main.py

---

📁 Data Flow
Collection → Social media posts collected (e.g., Reddit)

Preprocessing → Clean and tokenize text

Claim Extraction → Extract factual statements

Verification → Run through:

Knowledge Graph Lookup

Fact Check APIs

Semantic Matcher

Verdict Engine → Final truth score

Reporting → Output report, send alerts

---

📌 Technologies Used
Python

spaCy, NLTK, transformers

Wikidata API

MongoDB or Cassandra

React

---

🛠️ To-Do
 Integrate Twitter/X data collection

 Improve false positive handling in trend detection

 Add user feedback loop

 Deploy as Dockerized microservice

📜 License
This project is licensed under the MIT License.
