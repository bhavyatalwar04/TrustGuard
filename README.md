# 🛡️ TruthGuard

**TruthGuard** is a modular and extensible fact-checking and trend-detection pipeline designed for verifying claims made on social media platforms like Reddit. It uses NLP and knowledge graph lookups to detect misinformation, generate reports, and send alerts.

---

## 📂 Project Structure
TruthGuard/
├── data/ # Raw and processed post data
├── src/ # Core source code (data collection → processing → verification)
├── tests/ # Unit and integration tests
├── scripts/ # Utility scripts for setup/migration
├── frontend/ # (Optional) React-based frontend dashboard
├── requirements.txt # Python dependencies
└── README.md # You are here!
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