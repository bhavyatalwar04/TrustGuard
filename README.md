# TruthGuard

🔍 **Real-time misinformation detection and fact-checking platform**

TruthGuard is an AI-powered system designed to identify, verify, and track misinformation across digital platforms. It combines advanced natural language processing, knowledge graph technology, and real-time trend analysis to provide comprehensive fact-checking capabilities.

## 🌟 Features

- **Automated Claim Extraction**: Intelligently extracts factual claims from text content
- **Real-time Fact Checking**: Verifies claims against trusted knowledge sources
- **Semantic Matching**: Advanced similarity detection to identify related claims
- **Trend Detection**: Monitors misinformation patterns and emerging false narratives
- **Knowledge Graph Integration**: Maintains contextual relationships between facts and sources
- **Alert System**: Proactive notifications for high-risk misinformation
- **Interactive Dashboard**: Real-time visualization of verification results and trends
- **Report Generation**: Comprehensive analysis reports for stakeholders

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **API Layer**: RESTful endpoints for claim verification and data access
- **Services Layer**: Core business logic for fact-checking and analysis
- **Database Layer**: Persistent storage for claims, verifications, and user data
- **Models**: Data structures for claims, verifications, and users

### Frontend (React.js)
- **Dashboard**: Overview of system performance and recent verifications
- **Claim Checker**: Interactive interface for manual claim verification
- **Trend Analysis**: Visual representation of misinformation patterns
- **Report Viewer**: Display detailed verification reports
- **Alert Panel**: Real-time notifications and alerts

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+ (for local development)
- Node.js 16+ (for frontend development)

### Using Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/truthguard.git
cd truthguard

# Start all services
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Local Development Setup

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/setup_db.py

# Run the application
uvicorn app.main:app --reload --port 8000
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 📊 API Endpoints

### Claims
- `POST /api/claims/verify` - Verify a new claim
- `GET /api/claims/{id}` - Get claim details
- `GET /api/claims/` - List claims with filters

### Trends
- `GET /api/trends/` - Get trending misinformation patterns
- `GET /api/trends/analysis` - Detailed trend analysis

### Reports
- `GET /api/reports/` - Generate verification reports
- `GET /api/reports/{id}` - Get specific report

### Alerts
- `GET /api/alerts/` - Get active alerts
- `POST /api/alerts/subscribe` - Subscribe to alert notifications

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/truthguard

# API Keys
OPENAI_API_KEY=your_openai_key
GOOGLE_SEARCH_API_KEY=your_google_key
CUSTOM_SEARCH_ENGINE_ID=your_cse_id

# Security
SECRET_KEY=your_secret_key
ALGORITHM=HS256

# External Services
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📁 Project Structure

```
TruthGuard/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   ├── api/           # API routes and schemas
│   │   └── utils/         # Utility functions
│   ├── tests/             # Backend tests
│   ├── scripts/           # Database and migration scripts
│   └── data/              # Data storage directories
├── frontend/               # React.js frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API communication
│   │   ├── utils/         # Frontend utilities
│   │   └── styles/        # CSS styles
│   └── public/            # Static assets
└── docker-compose.yml      # Container orchestration
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint configuration for JavaScript
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-based text analysis
- FastAPI for the robust backend framework
- React.js community for frontend components
- Various fact-checking organizations for datasets and methodologies

## 📞 Support

- 📧 Email: support@truthguard.dev
- 💬 Discord: [TruthGuard Community](https://discord.gg/truthguard)
- 📖 Documentation: [docs.truthguard.dev](https://docs.truthguard.dev)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/truthguard/issues)

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Mobile application
- [ ] Browser extension
- [ ] Integration with social media platforms
- [ ] Advanced ML model training
- [ ] Blockchain-based verification logging

---

**⚠️ Disclaimer**: TruthGuard is a tool to assist in fact-checking and should not be the sole source for determining truth. Always verify important information through multiple trusted sources.