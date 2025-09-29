# Social Media Sentiment Analysis Platform

A comprehensive real-time social media sentiment analysis application built with Python, React, and machine learning. This platform allows you to analyze sentiment from social media feeds, compare multiple ML models, and visualize insights through an interactive dashboard.

## 🚀 Features

### Core Functionality
- **Real-time Twitter Stream Analysis** - Monitor live social media feeds with keyword filtering
- **Historical Tweet Search** - Search and analyze past tweets with comprehensive sentiment scoring
- **Multi-Model Sentiment Analysis** - Compare performance across 6+ different ML models:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - Neural Networks (Deep Learning)
  - LSTM (Recurrent Neural Networks)
  - Ensemble methods

### Advanced Features
- **A/B Testing Framework** - Compare model performance with statistical significance testing
- **Interactive Dashboards** - Real-time visualization of sentiment trends and metrics
- **Model Performance Analytics** - Detailed accuracy, precision, recall, and F1-score comparisons
- **Confidence Scoring** - Get confidence levels for each prediction
- **Text Preprocessing Pipeline** - Advanced NLP preprocessing with NLTK and spaCy

## 🏗️ Architecture

```
├── backend/
│   ├── api/                    # Flask API endpoints
│   ├── models/                 # ML models and training scripts
│   ├── utils/                  # Text processing and feature extraction
│   └── config.py              # Configuration management
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Application pages
│   │   └── services/         # API service layer
├── data/                      # Data storage
└── tests/                     # Test suites
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Twitter API credentials (Bearer Token, API Keys)
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SocialMediaAnalysis
```

### 2. Backend Setup

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Download Required NLP Models
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

#### Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` with your Twitter API credentials:
```env
TWITTER_BEARER_TOKEN=your_bearer_token_here
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET=your_api_secret_here
TWITTER_ACCESS_TOKEN=your_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret_here
FLASK_ENV=development
```

### 3. Frontend Setup

#### Install Node Dependencies
```bash
cd frontend
npm install
```

#### Configure Frontend Environment
Create `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:5000
```

## 🚀 Running the Application

### Start the Backend Server
```bash
python backend/api/flask_app.py
```
The API will be available at `http://localhost:5000`

### Start the Frontend Development Server
```bash
cd frontend
npm start
```
The web application will be available at `http://localhost:3000`

## 📖 Usage Guide

### 1. Real-time Sentiment Monitoring
1. Navigate to the "Real-time Dashboard" tab
2. Add keywords you want to monitor (e.g., "bitcoin", "#technology", "@username")
3. Click "Start Streaming" to begin real-time analysis
4. Watch live sentiment trends and tweet analysis

### 2. Historical Tweet Analysis
1. Go to the "Search & Analyze" tab
2. Enter search queries (keywords, hashtags, or usernames)
3. Set the number of tweets to analyze (max 100)
4. View comprehensive sentiment analysis results

### 3. Model Comparison & A/B Testing
1. Visit the "A/B Testing" tab
2. Create a new experiment with selected models
3. Run quick tests with sample data or upload your own dataset
4. Compare model performance metrics and statistical significance

## 🔧 API Endpoints

### Core Analytics
- `POST /api/search` - Search and analyze tweets
- `POST /api/stream/start` - Start real-time streaming
- `POST /api/stream/stop` - Stop streaming
- `GET /api/stream/data` - Get streaming data
- `POST /api/sentiment/analyze` - Analyze custom text

### A/B Testing
- `GET /api/ab-testing/experiments` - List experiments
- `POST /api/ab-testing/experiments` - Create experiment
- `POST /api/ab-testing/experiments/{id}/run` - Run experiment
- `GET /api/ab-testing/experiments/{id}/report` - Get detailed report

### Utilities
- `GET /api/trends` - Get trending topics
- `GET /api/user/{username}` - Analyze user's recent tweets
- `GET /api/health` - Health check

## 🧪 Model Training

### Training New Models
```python
from backend.models.sentiment_models import SentimentAnalyzer
from backend.utils.text_preprocessor import TextPreprocessor
from backend.utils.feature_extractor import FeatureExtractor

# Initialize components
analyzer = SentimentAnalyzer()
preprocessor = TextPreprocessor()
extractor = FeatureExtractor()
analyzer.load_dependencies(extractor, preprocessor)

# Train models with your data
texts = ["I love this!", "This is terrible", "It's okay"]
labels = [2, 0, 1]  # 0=negative, 1=neutral, 2=positive

analyzer.train_all_models(texts, labels)
analyzer.save_models()
```

### Model Performance Metrics
- **Accuracy** - Overall prediction correctness
- **Precision** - Quality of positive predictions
- **Recall** - Coverage of actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Confidence** - Model certainty in predictions

## 📊 Visualization Features

### Real-time Dashboard
- Live sentiment distribution (pie/bar charts)
- Time-series sentiment trends
- Tweet volume and engagement metrics
- Keyword performance tracking

### Analytics Dashboard
- Model comparison charts
- Confusion matrices
- Performance ranking tables
- Statistical significance tests

## 🔒 Security & Best Practices

- API keys are stored in environment variables
- Rate limiting implemented for Twitter API
- Input validation on all endpoints
- CORS properly configured
- No sensitive data logged

## 🧹 Development

### Code Style
- Python: Follow PEP 8 guidelines
- JavaScript: ESLint configuration included
- Components: Functional React components with hooks

### Testing
```bash
# Backend tests
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Twitter API Rate Limits**
- The application automatically handles rate limiting
- Consider upgrading to Twitter API v2 for higher limits

**Model Loading Errors**
- Ensure all required Python packages are installed
- Download spaCy models: `python -m spacy download en_core_web_sm`
- Check file permissions in the models directory

**Frontend Build Issues**
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`

**CORS Errors**
- Ensure backend is running on the correct port
- Check REACT_APP_API_URL in frontend/.env

### Performance Optimization

- Use ensemble models for best accuracy
- Use individual models (Naive Bayes, SVM) for faster predictions
- Implement caching for frequently accessed data
- Consider using Redis for real-time data storage

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the API documentation for endpoint details

## 🎯 Future Enhancements

- Support for additional social media platforms (Reddit, Instagram)
- Advanced sentiment categories (emotion detection)
- Multi-language sentiment analysis
- Real-time alerting system
- Data export capabilities (CSV, JSON, PDF reports)
- Integration with business intelligence tools
- Mobile application development