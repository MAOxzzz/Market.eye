# Market Eye AI 📈 - Stock Analysis System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100.0-green.svg" alt="FastAPI Version">
  <img src="https://img.shields.io/badge/Streamlit-1.28.0-red.svg" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg" alt="TensorFlow Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A powerful AI-powered stock analysis platform with LSTM forecasting, AI investment advice, and interactive visualization tools. The system provides stock price predictions, investment recommendations, and detailed PDF reports.

## 🌟 Features

- **Stock Price Visualization** - Interactive charts for historical stock data analysis
- **AI Investment Recommendations** - Get AI-powered BUY/HOLD/SELL advice
- **LSTM Price Forecasting** - Advanced deep learning models for price prediction
- **Enhanced PDF Report Generation** - Comprehensive reports with analytics, forecasts, error metrics, and AI-generated insights
- **User Authentication** - Secure login and activity tracking
- **Custom Dataset Upload** - Use your own stock datasets for analysis

## 📁 Project Structure

```
market-eye-ai/
├── backend/                  # Backend components
│   ├── agents/               # AI agents and intelligent components
│   ├── models/               # ML models and prediction algorithms
│   └── database/             # Database schemas and management
├── frontend/                 # Frontend components
│   └── pages/                # Individual UI pages and components
├── docs/                     # Documentation
│   └── report_templates/     # Templates for report generation
├── data/                     # Data directory
│   └── stock_data/           # Stock price datasets
├── forecasting/              # ML forecasting components
│   ├── models/               # Trained ML models
│   ├── data_updater.py       # Updates stock data
│   ├── forecast_api.py       # API for forecasting
│   └── stock_forecaster.py   # LSTM forecasting model
├── utils/                    # Utility functions and helpers
├── tests/                    # Test suite
├── scripts/                  # Utility scripts
├── reports/                  # Generated PDF reports
├── main.py                   # FastAPI backend
├── streamlit_app.py          # Streamlit frontend
├── requirements.txt          # Dependencies
├── CODE_OF_CONDUCT.md        # Code of conduct guidelines
└── CONTRIBUTING.md           # Contribution guidelines
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- SQLite3
- [Git](https://git-scm.com/)

### Installation

1. Clone the repository   
```bash
   git clone https://github.com/MAOxzzz/Market.eye.git
   cd Market.eye
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database
   ```bash
   python scripts/init_db.py
   ```

## 🔧 Running the Application

1. Start the backend server

   ```bash
   uvicorn main:app --reload
   ```

2. Start the Streamlit frontend (in a new terminal)

   ```bash
   streamlit run streamlit_app.py
   ```

3. Access the web app at http://localhost:8501

## 💻 Usage

1. **Login/Signup**: Create an account or log in to access all features
2. **View Stocks**: Visualize historical stock data for AAPL, MSFT, and GOOGL
3. **Get Advice**: Receive AI-generated investment recommendations
4. **Stock Forecast**: View price predictions for up to 90 days in the future
5. **Generate Reports**: Create and download comprehensive PDF reports

## 🧠 ML Models

The project uses LSTM (Long Short-Term Memory) neural networks to predict future stock prices:

- Sequence length: 30 days of historical data
- Architecture: LSTM(50) + Dense(1)
- Training: Adjustable epochs (default: 20) and batch size (default: 32)
- Evaluation: RMSE, MSE, R², and custom accuracy metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For detailed guidelines, please read our [CONTRIBUTING.md](CONTRIBUTING.md) file.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the interactive web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [TensorFlow](https://www.tensorflow.org/) for deep learning capabilities
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Google Generative AI](https://ai.google.dev/) for investment recommendations
