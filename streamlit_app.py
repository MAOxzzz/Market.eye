# streamlit_app.py

import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Market Eye AI", layout="wide")

import pandas as pd
from fpdf import FPDF
import base64
import os
import requests
import json
from datetime import datetime
import sys
import sqlite3

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import config settings 
import config

# Import the new PDF report generator
from forecasting.pdf_report_generator import PDFReportGenerator, generate_error_metrics_chart

# List to collect warning messages
warning_messages = []

# Import our forecasting API - this one should work as it's simplified
from forecasting.forecast_api import ForecastAPI

# Initialize flag variables
genai_available = False
crewai_available = False
langchain_available = False

# Try to import Google AI components, but handle import errors gracefully
try:
    import google.generativeai as genai
    genai_available = True
except ImportError:
    warning_messages.append("Google Generative AI module could not be imported. Some features will be disabled.")

# Try to import LangChain components separately to avoid the nested import error
try:
    # Only attempt to import langchain if Google Generative AI is available
    if genai_available:
        from langchain_google_genai import ChatGoogleGenerativeAI
        langchain_available = True
except (ImportError, AttributeError) as e:
    warning_messages.append(f"LangChain components could not be imported: {str(e)}. Some features will be disabled.")

# Try to import CrewAI components only if LangChain is available
try:
    # Only attempt to import CrewAI if both dependencies are available
    if genai_available and langchain_available:
        from agents.stock_analysis_crew import StockAnalysisCrew
        crewai_available = True
    else:
        warning_messages.append("Required dependencies for CrewAI are not available. Agent-based analysis will be disabled.")
except ImportError:
    warning_messages.append("CrewAI modules could not be imported. Agent-based analysis will be disabled.")

# Display collected warnings
for warning in warning_messages:
    st.warning(warning)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('market_eye.db')
    conn.row_factory = sqlite3.Row
    return conn

# Authentication logic
def authenticate_user(username, password):
    try:
        # Test if the server is available
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            server_available = response.status_code == 200
        except requests.exceptions.RequestException:
            # Server is not available
            st.error("Authentication server unavailable. Please start the backend server.")
            return False

        # If the server is available, proceed with normal authentication
        # Create a dummy email by appending @example.com to username
        dummy_email = f"{username}@example.com"
        
        response = requests.post(
            "http://127.0.0.1:8000/login",
            json={"username": username, "password": password, "email": dummy_email}
        )
        
        if response.status_code == 200:
            return True
        else:
            if response.status_code == 500:
                st.error("Authentication server error. Please try again later.")
            elif response.status_code == 422:
                st.error("Invalid login format.")
            else:
                st.error(f"Login failed with status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error connecting to backend server: {str(e)}")
        return False

def register_user(username, password, email):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/signup",
            json={"username": username, "password": password, "email": email}
        )
        
        # Return True only if the request was successful
        if response.status_code == 201:
            return True, "Account created successfully"
            
        # Handle specific error responses
        if response.status_code == 409:
            error_detail = response.json().get("detail", "")
            return False, f"Registration error: {error_detail}"
        elif response.status_code == 400:
            error_detail = response.json().get("detail", "")
            return False, f"Validation error: {error_detail}"
        else:
            return False, f"Server error: Status code {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, "Backend server is not running. Please start it with 'uvicorn main:app --reload'"
    except Exception as e:
        return False, f"Error occurred: {str(e)}"

# ----------------
# Configure Gemini AI 
# ----------------
if genai_available:
    # Use API key from config
    if config.GEMINI_API_KEY:
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            # Test the model to ensure it's working
            test_response = model.generate_content("Hello")
            if test_response:
                # Only show success message if we haven't shown it before in this session
                if "gemini_connected" not in st.session_state:
                    st.sidebar.success("✅ Gemini API connected successfully", icon="✅")
                    st.session_state.gemini_connected = True
            else:
                st.sidebar.warning("⚠️ Gemini API connection test failed. Recommendations will be disabled.")
                model = None
        except Exception as e:
            st.sidebar.error(f"⚠️ Error configuring Gemini API: {str(e)}")
            st.sidebar.info("💡 To fix: Create a .env file with your GEMINI_API_KEY.")
            model = None
    else:
        st.sidebar.warning("⚠️ Gemini API key not found in config. AI recommendations will be disabled.")
        st.sidebar.info("💡 To enable AI recommendations, create a .env file with your GEMINI_API_KEY.")
        model = None
else:
    st.sidebar.warning("⚠️ Google Generative AI module not available. Install with: pip install google-generativeai==0.3.2")
    model = None

# Load and clean our stock data
# Temporarily remove caching to debug the issue
def load_data():
    try:
        # Try to load the data
        df = pd.read_csv("data/stock_data/Dataset.csv")
        
        # Make sure the dataframe has the required columns
        required_columns = ['Date', 'Ticker', 'Close']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Required column '{col}' not found in dataset")
                return pd.DataFrame()
        
        # Clean up the data safely
        # Convert Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            
        # Convert Close to numeric
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Don't filter tickers - include all available tickers
        # if 'Ticker' in df.columns:
        #    df = df[df['Ticker'].isin(['AAPL', 'MSFT', 'GOOGL'])]
        
        # Drop any rows with NaN in critical columns
        df = df.dropna(subset=['Date', 'Close'])
        
        # Sort the data
        return df.sort_values(by=['Ticker', 'Date'])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Get forecast API
@st.cache_resource
def get_forecast_api():
    """Get or create the forecast API object."""
    return ForecastAPI()

# Replace the old generate_pdf_report function with a wrapper for our new module
def generate_pdf_report(recommendation, user, analytics_data=None, forecast_data=None, historical_data=None):
    """
    Generate a comprehensive PDF report using the PDFReportGenerator module.
    Returns an HTML link for downloading the report.
    """
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Initialize the PDF report generator
    report_generator = PDFReportGenerator(output_dir="reports")
    
    # Extract ticker from forecast data if available
    ticker = "STOCK"
    if forecast_data and 'ticker' in forecast_data:
        ticker = forecast_data['ticker']
    
    # Extract error metrics if available
    error_metrics = None
    if forecast_data and 'metrics' in forecast_data:
        error_metrics = forecast_data['metrics']
        
        # Generate error visualization if historical data is available
        if historical_data is not None and 'forecast_df' in forecast_data:
            try:
                # Get last N days of historical data that overlap with forecast
                n_days = min(10, len(forecast_data['forecast_df']))
                historical_tail = historical_data.tail(n_days)
                forecast_head = forecast_data['forecast_df'].head(n_days)
                
                # Generate error visualization
                if len(historical_tail) == len(forecast_head):
                    actual = historical_tail['Close'].values
                    predicted = forecast_head['Predicted_Close'].values
                    error_viz_buffer = generate_error_metrics_chart(actual, predicted, f"{ticker} - Forecast Error Analysis")
                    
                    # Save visualization to a temporary file
                    temp_viz_path = os.path.join("reports", f'temp_error_viz_{ticker}.png')
                    with open(temp_viz_path, 'wb') as f:
                        f.write(error_viz_buffer.getvalue())
                    
                    # Add visualization path to error metrics
                    error_metrics['visualization'] = temp_viz_path
            except Exception as e:
                st.warning(f"Could not generate error visualization: {str(e)}")
    
    # Generate the PDF report
    try:
        report_path = report_generator.create_analytical_report(
            user=user,
            ticker=ticker,
            analytics_data=analytics_data,
            forecast_data=forecast_data,
            historical_data=historical_data,
            recommendation=recommendation,
            error_metrics=error_metrics
        )
        
        # Generate download link
        download_link = report_generator.get_download_link(report_path)
        
        # Clean up temporary visualization file if it exists
        temp_viz_path = os.path.join("reports", f'temp_error_viz_{ticker}.png')
        if os.path.exists(temp_viz_path):
            os.remove(temp_viz_path)
        
        return download_link
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return "Error generating PDF report"

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    
if "username" not in st.session_state:
    st.session_state.username = ""

# Initialize CrewAI state
if "crew_analysis_results" not in st.session_state:
    st.session_state.crew_analysis_results = None

if "crew_analysis_running" not in st.session_state:
    st.session_state.crew_analysis_running = False

# Authentication UI
def auth_ui():
    st.title("Market Eye Authentication")
    
    # Toggle between login and signup
    auth_option = st.radio("Action", ["Log In", "Sign Up"])
    
    if auth_option == "Log In":
        st.header("Log In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Log In"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.header("Sign Up")
        new_username = st.text_input("Choose Username", help="3-20 characters, letters, numbers, underscore, hyphen only")
        new_email = st.text_input("Email Address", help="Enter a valid email address")
        new_password = st.text_input("Choose Password", type="password", help="Minimum 8 characters, must include uppercase, lowercase, and number")
        
        # Add input validation
        validation_error = None
        
        # Validate on button click
        if st.button("Sign Up"):
            # Validate username
            if not new_username or len(new_username) < 3 or len(new_username) > 20:
                validation_error = "Username must be 3-20 characters"
            elif not all(c.isalnum() or c in "_-" for c in new_username):
                validation_error = "Username can only contain letters, numbers, underscore and hyphen"
            
            # Validate email
            elif not new_email or "@" not in new_email or "." not in new_email:
                validation_error = "Please enter a valid email address"
            
            # Validate password
            elif not new_password or len(new_password) < 8:
                validation_error = "Password must be at least 8 characters"
            elif not any(c.isupper() for c in new_password):
                validation_error = "Password must contain at least one uppercase letter"
            elif not any(c.islower() for c in new_password):
                validation_error = "Password must contain at least one lowercase letter"
            elif not any(c.isdigit() for c in new_password):
                validation_error = "Password must contain at least one number"
            
            # If validation passes, attempt registration
            if validation_error:
                st.error(validation_error)
            else:
                success, message = register_user(new_username, new_password, new_email)
                if success:
                    st.success(message)
                else:
                    st.error(message)

# Main app UI after login
def main_app():
    # Load data
    df = load_data()
    
    st.title(f"Welcome to Market Eye, {st.session_state.username}!")
    st.write("Your AI-powered stock analysis platform")
    
    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    
    # Create tabs - includes CrewAI Analysis tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["View Stocks", "Get Advice", "Stock Forecast", "CrewAI Analysis", "Dataset Management", "Save Report"])
    
    # TAB 1: Stock Viewer
    with tab1:
        st.header("View Stock Data")
        
        # Let user pick which stock to view
        df = load_data()
        
        # Create a more informative ticker display with company name when available
        tickers = df['Ticker'].unique()
        ticker_options = []
        
        # Create nicer display names for the dropdown with company names when available
        for ticker in sorted(tickers):
            ticker_rows = df[df['Ticker'] == ticker]
            if 'Brand_Name' in df.columns and not ticker_rows['Brand_Name'].isna().all():
                brand_name = ticker_rows['Brand_Name'].iloc[0]
                if isinstance(brand_name, str) and brand_name.strip():
                    ticker_options.append(f"{ticker} - {brand_name.title()}")
                else:
                    ticker_options.append(ticker)
            else:
                ticker_options.append(ticker)
        
        selected_option = st.selectbox("Choose a stock:", ticker_options)
        
        # Extract the ticker from the selected option
        selected_stock = selected_option.split(" - ")[0] if " - " in selected_option else selected_option
        
        # Show the stock's price chart
        stock_data = df[df['Ticker'] == selected_stock]
        st.line_chart(stock_data.set_index('Date')['Close'])
        
        # Show recent data
        st.write("Recent prices:")
        st.dataframe(stock_data.tail())
    
    # TAB 2: AI Advisor
    with tab2:
        st.header("Get AI Investment Advice")
        
        # Load the data first (in case it wasn't loaded properly) 
        df = load_data()
        
        if df.empty:
            st.error("Unable to load stock data. Please check your dataset.")
        else:
            # Create ticker options with company names
            tickers = df['Ticker'].unique()
            ticker_options = []
            
            # Create nicer display names for the dropdown with company names when available
            for ticker in sorted(tickers):
                ticker_rows = df[df['Ticker'] == ticker]
                if 'Brand_Name' in df.columns and not ticker_rows['Brand_Name'].isna().all():
                    brand_name = ticker_rows['Brand_Name'].iloc[0]
                    if isinstance(brand_name, str) and brand_name.strip():
                        ticker_options.append(f"{ticker} - {brand_name.title()}")
                    else:
                        ticker_options.append(ticker)
                else:
                    ticker_options.append(ticker)
            
            selected_option = st.selectbox("Select stock:", ticker_options)
            
            # Extract the ticker from the selected option
            stock = selected_option.split(" - ")[0] if " - " in selected_option else selected_option
            
            # Get date range with error handling
            try:
                min_date = df['Date'].min().date()
                max_date = df['Date'].max().date()
            except:
                # Fallback if dates can't be extracted
                min_date = datetime(2020, 1, 1).date()
                max_date = datetime(2023, 12, 31).date()
            
            # Display the date ranges with reasonable defaults
            st.write(f"Available data range: {min_date} to {max_date}")
            start_date = st.date_input("Start date", min_date)
            end_date = st.date_input("End date", max_date)
            
            if st.button("Get Recommendation"):
                # Make sure dates are in the correct format
                try:
                    # Convert input dates to string format for safe comparison
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')
                    
                    # Filter data manually to avoid timezone issues
                    filtered_data = []
                    for _, row in df.iterrows():
                        try:
                            row_date_str = row['Date'].strftime('%Y-%m-%d')
                            if (row['Ticker'] == stock and 
                                row_date_str >= start_str and 
                                row_date_str <= end_str):
                                filtered_data.append(row)
                        except:
                            continue
                    
                    # Convert back to DataFrame
                    period_data = pd.DataFrame(filtered_data)
                    
                    if len(period_data) < 2:
                        st.warning("Please select a longer time period. Not enough data points found.")
                    else:
                        # Calculate price change and percentage change
                        price_change = period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]
                        percent_change = (price_change/period_data['Close'].iloc[0]) * 100
                        
                        # Show basic info
                        st.subheader(f"{stock} Performance")
                        st.write(f"Price change: ${price_change:.2f} ({percent_change:.2f}%)")
                        st.write(f"Data points: {len(period_data)}")
                        
                        # Get AI recommendation
                        with st.spinner("Getting AI advice..."):
                            prompt = f"""
                            Give simple investment advice for {stock} stock based on this data:
                            Time period: {start_date} to {end_date}
                            Price change: {percent_change:.2f}%
                            Current price: ${period_data['Close'].iloc[-1]:.2f}
                            
                            Recommend either BUY, HOLD, or SELL in the first line.
                            Then explain your reasoning in 2-3 simple sentences.
                            """
                            
                            if genai_available and model is not None:
                                response = model.generate_content(prompt)
                                st.subheader("AI Recommendation")
                                st.write(response.text)
                                
                                # Save for PDF report
                                st.session_state.recommendation = response.text
                            else:
                                # Fallback recommendation when Gemini is not available
                                st.subheader("AI Recommendation")
                                fallback_text = f"""HOLD

The AI recommendation feature is currently unavailable.

Possible solutions:
1. Check that you have a valid Gemini API key in the .env file
2. Make sure you have installed the required packages:
   pip install google-generativeai==0.3.2 langchain-google-genai==0.0.3
3. Check the sidebar for specific error details

Based on the data shown above for {stock}, you can evaluate the price trends of ${price_change:.2f} ({percent_change:.2f}%) to make your own investment decision."""
                                st.write(fallback_text)
                                
                                # Save for PDF report
                                st.session_state.recommendation = fallback_text
                            
                            # Log this activity in the database
                            try:
                                conn = get_db_connection()
                                cursor = conn.cursor()
                                user_id = cursor.execute("SELECT user_id FROM users WHERE username = ?", 
                                                    (st.session_state.username,)).fetchone()['user_id']
                                cursor.execute("INSERT INTO activity_logs (user_id, action) VALUES (?, ?)",
                                            (user_id, f"Generated recommendation for {stock}"))
                                conn.commit()
                                conn.close()
                            except Exception as e:
                                st.error(f"Could not log activity: {e}")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.write("Please try a different date range or stock.")

    # TAB 3: Stock Forecaster
    with tab3:
        st.header("Stock Price Forecaster")
        
        # Initialize forecast API
        forecast_api = get_forecast_api()
        
        # Get available tickers with models
        available_tickers = forecast_api.get_available_tickers()
        
        # Create tabs for different forecasting features
        forecast_tab1, forecast_tab2, forecast_tab3 = st.tabs(["Quick Forecast", "Analytics", "Advanced Forecast"])
        
        # TAB 1: Quick Forecast
        with forecast_tab1:
            st.subheader("Quick Stock Price Forecast")
            
            if not available_tickers:
                st.warning("No forecasting models available. Please use the Analytics tab to analyze stock data.")
            else:
                # Select ticker
                ticker = st.selectbox(
                    "Select a stock to forecast:",
                    options=available_tickers,
                    key="forecast_ticker"
                )
                
                # Select forecast period
                forecast_days = st.slider(
                    "Number of days to forecast:",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=1
                )
                
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        try:
                            # Get historical and forecast data
                            historical_df, forecast_df = forecast_api.get_historical_and_forecast(
                                ticker, days=forecast_days, historical_days=90
                            )
                            
                            if historical_df is not None and forecast_df is not None:
                                # Display the forecast
                                st.success(f"Forecast generated for {ticker} for the next {forecast_days} days")
                                
                                # Create chart
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                fig = make_subplots(specs=[[{"secondary_y": False}]])
                                
                                # Add historical prices
                                fig.add_trace(
                                    go.Scatter(
                                        x=historical_df['Date'],
                                        y=historical_df['Close'],
                                        mode='lines',
                                        name='Historical',
                                        line=dict(color='blue')
                                    )
                                )
                                
                                # Add forecasted prices
                                fig.add_trace(
                                    go.Scatter(
                                        x=forecast_df['Date'],
                                        y=forecast_df['Predicted_Close'],
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='red', dash='dash')
                                    )
                                )
                                
                                # Add chart details
                                fig.update_layout(
                                    title=f"{ticker} Stock Price Forecast",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                
                                # Show chart
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display metrics
                                metrics = forecast_api.get_forecast_metrics(ticker)
                                
                                # Create columns for metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                                with col2:
                                    st.metric("MSE", f"{metrics['mse']:.2f}")
                                with col3:
                                    st.metric("R²", f"{metrics['r2']:.2f}")
                                with col4:
                                    st.metric("Accuracy (±5%)", f"{metrics['accuracy_5pct']:.1f}%")
                                
                                # Display forecast data
                                with st.expander("View Forecast Data"):
                                    st.dataframe(forecast_df)
                                    
                                # Save forecast for PDF report
                                st.session_state.forecast_data = {
                                    'ticker': ticker,
                                    'historical_df': historical_df,
                                    'forecast_df': forecast_df,
                                    'metrics': metrics
                                }
                                
                                # Add download button for forecast data
                                csv = forecast_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Forecast CSV",
                                    data=csv,
                                    file_name=f"{ticker}_forecast.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error(f"Could not generate forecast for {ticker}. Not enough data.")
                        except Exception as e:
                            st.error(f"Error generating forecast: {str(e)}")
            
            # Enhanced dataset option
            if not available_tickers:
                enhanced_dataset = 'data/stock_data/enhanced_dataset.csv'
                original_dataset = 'data/stock_data/Dataset.csv'
                
                # Check if enhanced dataset exists
                enhanced_exists = os.path.exists(enhanced_dataset)
                
                # Allow users to upload their own dataset
                st.subheader("Dataset Selection")
                upload_tab, existing_tab = st.tabs(["Upload New Dataset", "Use Existing Dataset"])
                
                with upload_tab:
                    st.write("Upload your own stock price dataset (CSV format)")
                    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
                    
                    if uploaded_file is not None:
                        try:
                            # Save the uploaded file
                            upload_path = 'data/stock_data/uploaded_dataset.csv'
                            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
                            
                            with open(upload_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Preview the data
                            df = pd.read_csv(upload_path)
                            st.write(f"Successfully uploaded dataset with {len(df)} rows")
                            st.dataframe(df.head())
                            
                            # Set as current dataset
                            dataset_to_check = upload_path
                            st.success("✅ Dataset uploaded successfully! You can now check or train with this dataset.")
                        except Exception as e:
                            st.error(f"Error uploading file: {str(e)}")
                
                with existing_tab:
                    if enhanced_exists:
                        st.info("An enhanced dataset with more historical data is available.")
                        dataset_to_check = enhanced_dataset
                    else:
                        st.info("Using the default dataset.")
                        dataset_to_check = original_dataset
                    
                    # Check if default dataset exists
                    if os.path.exists(dataset_to_check):
                        st.success(f"✅ Using dataset: {os.path.basename(dataset_to_check)}")
                        # Preview the data
                        try:
                            df = pd.read_csv(dataset_to_check)
                            st.write(f"Dataset contains {len(df)} rows")
                            st.dataframe(df.head())
                        except Exception as e:
                            st.error(f"Error reading dataset: {str(e)}")
                    else:
                        st.error(f"❌ Dataset file not found: {dataset_to_check}")
        
        # TAB 2: Analytics
        with forecast_tab2:
            st.subheader("Stock Analytics")
            
            try:
                # Try to import StockAnalyzer
                from forecasting.stock_analyzer import StockAnalyzer
                
                # Select data source
                data_sources = {
                    'Dataset.csv': 'data/stock_data/Dataset.csv',
                    'Enhanced Dataset': 'data/stock_data/enhanced_dataset.csv',
                    'Uploaded Dataset': 'data/stock_data/uploaded_dataset.csv',
                    'Updated Dataset': 'data/stock_data/updated_dataset.csv'
                }
                
                # Filter to only existing files
                available_sources = {k: v for k, v in data_sources.items() if os.path.exists(v)}
                
                if not available_sources:
                    st.error("No data sources found. Please upload or create a dataset.")
                else:
                    # Select data source
                    selected_source = st.selectbox(
                        "Select data source:",
                        options=list(available_sources.keys()),
                        key="analytics_data_source"
                    )
                    
                    data_path = available_sources[selected_source]
                    
                    # Initialize analyzer
                    analyzer = StockAnalyzer(data_path, output_dir='forecasting/analysis')
                    
                    # Load data
                    with st.spinner("Loading data..."):
                        analyzer.load_data()
                    
                    # Select tickers for analysis
                    all_tickers = sorted(analyzer.df['Ticker'].unique().tolist())
                    
                    selected_tickers = st.multiselect(
                        "Select stocks to analyze:",
                        options=all_tickers,
                        default=all_tickers[:3] if len(all_tickers) >= 3 else all_tickers[:1],
                        key="analytics_tickers"
                    )
                    
                    if selected_tickers:
                        if st.button("Run Analysis"):
                            with st.spinner("Analyzing stock data..."):
                                # Generate analytics
                                company_comparison = analyzer.generate_cross_company_comparison(selected_tickers)
                                
                                # Display company comparison
                                st.subheader("Company Comparison")
                                st.dataframe(company_comparison)
                                
                                # Plot comparison charts
                                analyzer.plot_comparison_charts(selected_tickers)
                                
                                # Display charts
                                price_chart_path = os.path.join(analyzer.output_dir, 'price_comparison.png')
                                growth_chart_path = os.path.join(analyzer.output_dir, 'growth_comparison.png')
                                
                                if os.path.exists(price_chart_path) and os.path.exists(growth_chart_path):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.image(price_chart_path, caption="Price Comparison")
                                    
                                    with col2:
                                        st.image(growth_chart_path, caption="Growth Comparison")
                                
                                # Generate sector comparison
                                sector_comparison = analyzer.generate_sector_comparison()
                                
                                # Display sector comparison
                                st.subheader("Sector Comparison")
                                st.dataframe(sector_comparison)
                                
                                # Save analytics for PDF report
                                st.session_state.analytics_data = {
                                    'company_comparison': company_comparison,
                                    'sector_comparison': sector_comparison,
                                    'selected_tickers': selected_tickers
                                }
                                
                                # Add download button for analytics data
                                csv = company_comparison.to_csv(index=False)
                                st.download_button(
                                    label="Download Analytics CSV",
                                    data=csv,
                                    file_name="company_comparison.csv",
                                    mime="text/csv"
                                )
            except ImportError:
                st.error("StockAnalyzer module could not be imported. Analytics functionality is disabled.")
        
        # TAB 3: Advanced Forecast
        with forecast_tab3:
            st.subheader("Advanced Stock Forecasting")
            
            try:
                # Try to import LSTMForecaster
                from forecasting.lstm_forecaster import LSTMForecaster
                
                # Select data source
                data_sources = {
                    'Dataset.csv': 'data/stock_data/Dataset.csv',
                    'Enhanced Dataset': 'data/stock_data/enhanced_dataset.csv',
                    'Uploaded Dataset': 'data/stock_data/uploaded_dataset.csv',
                    'Updated Dataset': 'data/stock_data/updated_dataset.csv'
                }
                
                # Filter to only existing files
                available_sources = {k: v for k, v in data_sources.items() if os.path.exists(v)}
                
                if not available_sources:
                    st.error("No data sources found. Please upload or create a dataset.")
                else:
                    # Select data source
                    selected_source = st.selectbox(
                        "Select data source:",
                        options=list(available_sources.keys()),
                        key="advanced_data_source"
                    )
                    
                    data_path = available_sources[selected_source]
                    
                    # Initialize forecaster
                    forecaster = LSTMForecaster(data_path, model_dir='forecasting/models')
                    
                    # Create tabs for training and forecasting
                    lstm_tab1, lstm_tab2 = st.tabs(["Train Model", "Generate January 2025 Forecast"])
                    
                    # Training tab
                    with lstm_tab1:
                        st.subheader("Train LSTM Model")
                        
                        # Load data
                        with st.spinner("Loading data..."):
                            forecaster.load_data()
                        
                        # Get available tickers
                        all_tickers = sorted(forecaster.df['Ticker'].unique().tolist())
                        
                        # Select ticker for training
                        ticker = st.selectbox(
                            "Select stock to train model for:",
                            options=all_tickers,
                            key="lstm_train_ticker"
                        )
                        
                        # Training parameters
                        epochs = st.slider("Number of epochs:", min_value=10, max_value=100, value=20, step=5)
                        batch_size = st.slider("Batch size:", min_value=8, max_value=64, value=32, step=8)
                        
                        if st.button("Train Model"):
                            with st.spinner(f"Training LSTM model for {ticker}... This may take a few minutes."):
                                try:
                                    # Train model
                                    training_info = forecaster.train_model(ticker, epochs=epochs, batch_size=batch_size)
                                    
                                    # Display training info
                                    st.success(f"Model trained successfully: {training_info['message']}")
                                    st.write(f"Epochs trained: {training_info['epochs_trained']}")
                                    st.write(f"Final loss: {training_info['final_loss']:.6f}")
                                    st.write(f"Final validation loss: {training_info['final_val_loss']:.6f}")
                                    
                                    # Evaluate model
                                    st.subheader("Model Evaluation")
                                    evaluation = forecaster.evaluate_model(ticker)
                                    
                                    # Display metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("MSE", f"{evaluation['mse']:.2f}")
                                    with col2:
                                        st.metric("RMSE", f"{evaluation['rmse']:.2f}")
                                    with col3:
                                        st.metric("MAE", f"{evaluation['mae']:.2f}")
                                    with col4:
                                        st.metric("Error %", f"{evaluation['percentage_error']:.2f}%")
                                    
                                    # Display evaluation plot
                                    if os.path.exists(evaluation['plot_path']):
                                        st.image(evaluation['plot_path'], caption="Model Evaluation")
                                except Exception as e:
                                    st.error(f"Error training model: {str(e)}")
                    
                    # Forecasting tab
                    with lstm_tab2:
                        st.subheader("January 2025 Forecast")
                        
                        # Load data
                        with st.spinner("Loading data..."):
                            forecaster.load_data()
                        
                        # Get available tickers
                        all_tickers = sorted(forecaster.df['Ticker'].unique().tolist())
                        
                        # Select ticker for forecasting
                        ticker = st.selectbox(
                            "Select stock to forecast for January 2025:",
                            options=all_tickers,
                            key="lstm_forecast_ticker"
                        )
                        
                        if st.button("Generate January 2025 Forecast"):
                            with st.spinner(f"Generating January 2025 forecast for {ticker}..."):
                                try:
                                    # Check if model exists
                                    model_path = os.path.join(forecaster.model_dir, f'{ticker}_lstm_model.h5')
                                    
                                    if not os.path.exists(model_path):
                                        st.warning(f"No trained model found for {ticker}. Please train a model first.")
                                    else:
                                        # Generate forecast
                                        forecast_df = forecaster.forecast_january_2025(ticker)
                                        
                                        # Display forecast
                                        st.success(f"January 2025 forecast generated for {ticker}")
                                        
                                        # Display forecast plot
                                        plot_path = os.path.join(forecaster.model_dir, f'{ticker}_jan_2025_forecast.png')
                                        
                                        if os.path.exists(plot_path):
                                            st.image(plot_path, caption=f"{ticker} - January 2025 Forecast")
                                        
                                        # Display forecast data
                                        st.subheader("Forecast Data")
                                        st.dataframe(forecast_df)
                                        
                                        # Save forecast for PDF report
                                        st.session_state.lstm_forecast = {
                                            'ticker': ticker,
                                            'forecast_df': forecast_df
                                        }
                                        
                                        # Add download button for forecast data
                                        csv = forecast_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download January 2025 Forecast CSV",
                                            data=csv,
                                            file_name=f"{ticker}_jan_2025_forecast.csv",
                                            mime="text/csv"
                                        )
                                except Exception as e:
                                    st.error(f"Error generating forecast: {str(e)}")
            except ImportError:
                st.error("LSTMForecaster module could not be imported. Advanced forecasting is disabled. Please install TensorFlow and Keras.")
                st.info("You can still use the Quick Forecast tab which uses a simpler model that doesn't require TensorFlow/Keras.")

    # TAB 4: CrewAI Analysis
    with tab4:
        st.header("AI-Powered Investment Analysis")
        
        # User input for tickers to analyze
        st.write("Select stocks for our AI crew to analyze")
        
        # Get available tickers from loaded data
        df = load_data()
        available_tickers = sorted(df['Ticker'].unique().tolist())
        
        # Multi-select for tickers
        selected_tickers = st.multiselect(
            "Select stocks to analyze (max 5):",
            options=available_tickers,
            default=available_tickers[:3] if len(available_tickers) >= 3 else available_tickers[:1],
            max_selections=5
        )
        
        # Option to include sector comparison
        include_sector = st.checkbox("Include sector comparison", value=True)
        
        # Show explanation of the process
        with st.expander("How does this work?"):
            st.write("""
            This analysis uses an AI crew with specialized agents to analyze your selected stocks:
            
            1. **Data Collector Agent**: Gathers and prepares stock data
            2. **Data Processor Agent**: Analyzes trends and generates forecasts
            3. **LLM Recommendation Agent**: Creates investment advice using Google's Gemini API
            
            The process may take a few minutes to complete.
            """)
        
        # Button to run the analysis
        if st.button("Run CrewAI Analysis"):
            if not selected_tickers:
                st.error("Please select at least one stock to analyze.")
            elif not crewai_available:
                st.error("CrewAI modules could not be loaded. This feature is disabled due to dependency issues.")
            else:
                # Show a spinner while the analysis is running
                with st.spinner("Running AI crew analysis... This may take a few minutes."):
                    try:
                        # Initialize the crew
                        crew = StockAnalysisCrew()
                        
                        # Store in session state that we're running analysis
                        st.session_state.crew_analysis_running = True
                        
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        st.info("Starting analysis...")
                        
                        # Run the analysis
                        results = crew.analyze_stocks(selected_tickers)
                        progress_bar.progress(100)
                        
                        # Store the results in session state
                        st.session_state.crew_analysis_results = results
                        st.session_state.crew_analysis_running = False
                        
                        # Show a success message
                        st.success("Analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running analysis: {str(e)}")
                        st.session_state.crew_analysis_running = False
                        
                        # Provide more detailed error information
                        with st.expander("Detailed Error Information (for debugging)"):
                            st.write(f"Error type: {type(e).__name__}")
                            st.write(f"Error message: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        # Display results if available
        if st.session_state.crew_analysis_results:
            results = st.session_state.crew_analysis_results
            
            # Display the recommendations
            st.subheader("Investment Recommendations")
            
            # Create columns for each ticker recommendation
            if 'recommendations' in results and 'ticker_recommendations' in results['recommendations']:
                recommendations = results['recommendations']['ticker_recommendations']
                
                # Display each recommendation in a separate expander
                for ticker, rec in recommendations.items():
                    with st.expander(f"{ticker} - {rec.get('recommendation_type', 'UNKNOWN')}"):
                        st.write(rec.get('recommendation_text', 'No recommendation available'))
                
                # Display market summary
                if 'market_summary' in results['recommendations']:
                    with st.expander("Market Summary"):
                        st.write(results['recommendations']['market_summary'])
                
                # Display analysis details
                if 'analysis' in results and 'metrics' in results['analysis']:
                    with st.expander("Analysis Details"):
                        # Create a DataFrame from the metrics
                        metrics = results['analysis']['metrics']
                        metrics_data = []
                        
                        for ticker, metric in metrics.items():
                            metrics_data.append({
                                'Ticker': ticker,
                                'Highest Price': f"${metric.get('highest_price', 'N/A')}",
                                'Lowest Price': f"${metric.get('lowest_price', 'N/A')}",
                                '2020 Growth': f"{metric.get('annual_growth_2020', 'N/A')}%"
                            })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df)
                
                # Display sector comparison if available
                if include_sector and 'analysis' in results and 'sector_comparison' in results['analysis']:
                    with st.expander("Sector Comparison"):
                        sector_comparison = results['analysis']['sector_comparison']
                        
                        for sector, data in sector_comparison.items():
                            st.write(f"### {sector} Sector")
                            st.write(f"Overall Growth: {data.get('overall_growth_pct', 'N/A')}%")
                            st.write(f"Average Volatility: {data.get('volatility', 'N/A')}")
                            st.write(f"Companies: {', '.join(data.get('tickers', []))}")
                            st.write("---")

    # TAB 5: Dataset Management
    with tab5:
        st.header("Dataset Management")
        
        # Import necessary modules
        from forecasting.data_updater import update_dataset, backfill_dataset
        import subprocess
        import datetime
        
        # Show dataset info
        st.subheader("Current Dataset Information")
        
        # Determine source dataset (in order of preference)
        source_datasets = [
            'data/stock_data/enhanced_dataset.csv',
            'data/stock_data/uploaded_dataset.csv',
            'data/stock_data/Dataset.csv',
            'data/stock_data/updated_dataset.csv'
        ]
        
        current_dataset = None
        for dataset in source_datasets:
            if os.path.exists(dataset):
                current_dataset = dataset
                break
        
        if current_dataset:
            try:
                df = pd.read_csv(current_dataset)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
                
                # Display dataset info
                st.write(f"**Dataset Path:** {current_dataset}")
                st.write(f"**Total Rows:** {len(df):,}")
                st.write(f"**Unique Tickers:** {len(df['Ticker'].unique())}")
                
                try:
                    date_range = [
                        df['Date'].min().strftime('%Y-%m-%d'),
                        df['Date'].max().strftime('%Y-%m-%d')
                    ]
                    st.write(f"**Date Range:** {date_range[0]} to {date_range[1]}")
                except:
                    st.write("**Date Range:** Unable to determine")
                
                st.write("**Sample Data:**")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error reading dataset: {str(e)}")
        else:
            st.error("No dataset found. Please upload or create one.")
        
        # Dataset update options
        st.subheader("Update Options")
        
        update_tab1, update_tab2, update_tab3 = st.tabs(["One-Time Update", "Backfill Data", "Schedule Updates"])
        
        # TAB 1: One-Time Update
        with update_tab1:
            st.write("Update the dataset with one day of new data.")
            
            if st.button("Run One-Time Update", key="onetime_update"):
                try:
                    with st.spinner("Updating dataset..."):
                        output_path = 'data/stock_data/updated_dataset.csv'
                        result_df = update_dataset(current_dataset, output_path)
                        
                        # Show result summary
                        st.success(f"Dataset updated successfully. Now contains {len(result_df):,} rows.")
                        st.write(f"Latest date: {result_df['Date'].max().strftime('%Y-%m-%d')}")
                except Exception as e:
                    st.error(f"Error updating dataset: {str(e)}")
        
        # TAB 2: Backfill Data
        with update_tab2:
            st.write("Backfill the dataset with simulated data up to a specific date.")
            
            # Date selector for the end date
            today = datetime.datetime.now().date()
            default_date = today + datetime.timedelta(days=60)
            end_date = st.date_input(
                "End Date for Backfilling", 
                value=default_date,
                min_value=today,
                max_value=datetime.datetime(2025, 12, 31).date()
            )
            
            if st.button("Run Backfill", key="backfill"):
                try:
                    with st.spinner(f"Backfilling dataset to {end_date}..."):
                        output_path = 'data/stock_data/updated_dataset.csv'
                        # Ensure we pass a timezone-naive date string
                        end_date_str = end_date.strftime('%Y-%m-%d')
                        result_df = backfill_dataset(
                            current_dataset, 
                            end_date=end_date_str, 
                            output_path=output_path
                        )
                        
                        # Show result summary
                        st.success(f"Dataset backfilled successfully. Now contains {len(result_df):,} rows.")
                        st.write(f"Date range: {result_df['Date'].min().strftime('%Y-%m-%d')} to {result_df['Date'].max().strftime('%Y-%m-%d')}")
                        
                        # Display the backfilled data
                        with st.expander("View Backfilled Data"):
                            st.dataframe(result_df.tail())
                except Exception as e:
                    st.error(f"Error backfilling dataset: {str(e)}")
        
        # TAB 3: Schedule Updates
        with update_tab3:
            st.write("Set up scheduled daily updates until a specified end date.")
            
            # Date and time selectors
            update_time = st.time_input("Daily Update Time", value=datetime.time(0, 0))
            
            today = datetime.datetime.now().date()
            default_date = today + datetime.timedelta(days=60)
            end_date = st.date_input(
                "End Date for Updates", 
                value=default_date,
                min_value=today,
                max_value=datetime.datetime(2025, 12, 31).date()
            )
            
            # Windows-specific
            if os.name == 'nt':
                st.write("#### Windows Task Scheduler Setup")
                
                if st.button("Set Up Windows Scheduled Tasks"):
                    try:
                        with st.spinner("Setting up scheduled tasks..."):
                            # Get the project directory
                            project_dir = os.path.abspath(".")
                            
                            # Format dates as strings to avoid timezone issues
                            end_date_str = end_date.strftime('%Y-%m-%d')
                            update_time_str = update_time.strftime('%H:%M')
                            
                            # Set up daily update task
                            update_cmd = f"schtasks /create /tn \"MarketEyeAI_Stock_Update\" /tr \"python {project_dir}\\forecasting\\data_updater.py --update\" /sc DAILY /st {update_time_str} /f"
                            
                            # Set up monthly backfill task
                            backfill_cmd = f"schtasks /create /tn \"MarketEyeAI_Stock_Backfill\" /tr \"python {project_dir}\\forecasting\\data_updater.py --backfill --end-date {end_date_str}\" /sc MONTHLY /mo 1 /st {(datetime.datetime.combine(datetime.date.today(), update_time) + datetime.timedelta(hours=1)).time().strftime('%H:%M')} /f"
                            
                            # Run the commands
                            subprocess.run(update_cmd, shell=True, check=True)
                            subprocess.run(backfill_cmd, shell=True, check=True)
                            
                            st.success("Scheduled tasks set up successfully!")
                            st.write("Tasks created:")
                            st.write(f"- MarketEyeAI_Stock_Update: Runs daily at {update_time.strftime('%H:%M')}")
                            st.write(f"- MarketEyeAI_Stock_Backfill: Runs monthly to ensure backfilling until {end_date.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        st.error(f"Error setting up scheduled tasks: {str(e)}")
            else:
                # For non-Windows systems, offer instructions
                st.write("#### Unix/Linux/Mac Scheduler Setup")
                
                # Generate the crontab entries
                daily_update = f"{update_time.minute} {update_time.hour} * * * cd {os.path.abspath('.')} && python -m forecasting.data_updater --update"
                monthly_backfill = f"{update_time.minute} {(update_time.hour + 1) % 24} 1 * * cd {os.path.abspath('.')} && python -m forecasting.data_updater --backfill --end-date {end_date.strftime('%Y-%m-%d')}"
                
                st.write("Add these lines to your crontab (run `crontab -e`):")
                st.code(daily_update)
                st.code(monthly_backfill)
                
                st.info("These commands will set up a daily update and a monthly backfill task.")
        
        # Data verification section
        st.subheader("Verify Updated Data")
        
        if os.path.exists('data/stock_data/updated_dataset.csv'):
            if st.button("View Updated Dataset"):
                try:
                    updated_df = pd.read_csv('data/stock_data/updated_dataset.csv')
                    updated_df['Date'] = pd.to_datetime(updated_df['Date'], errors='coerce', utc=True)
                    
                    # Show statistics
                    st.write(f"**Total Rows:** {len(updated_df):,}")
                    st.write(f"**Unique Tickers:** {len(updated_df['Ticker'].unique())}")
                    st.write(f"**Date Range:** {updated_df['Date'].min().strftime('%Y-%m-%d')} to {updated_df['Date'].max().strftime('%Y-%m-%d')}")
                    
                    # Show the data for the most recent dates
                    st.write("**Most Recent Data:**")
                    recent_dates = updated_df['Date'].sort_values(ascending=False).unique()[:5]
                    recent_data = updated_df[updated_df['Date'].isin(recent_dates)].sort_values(['Date', 'Ticker'], ascending=[False, True])
                    st.dataframe(recent_data)
                except Exception as e:
                    st.error(f"Error reading updated dataset: {str(e)}")
        else:
            st.info("No updated dataset found. Run an update first.")

    # TAB 6: Save Report
    with tab6:
        st.header("Save Your Report")
        
        # Create tabs for different report types
        report_type = st.radio("Report Type", ["Stock Analysis", "Forecast Report", "CrewAI Analysis"], horizontal=True)
        
        if report_type == "Stock Analysis":
            # Check if we have analytics data to save
            analytics_data = st.session_state.get('analytics_data', None)
            recommendation = st.session_state.get('recommendation', None)
            
            if analytics_data or recommendation:
                st.write("Your stock analysis report is ready to save!")
                
                # Generate the stock analysis PDF report
                try:
                    report_generator = PDFReportGenerator(output_dir="reports")
                    report_path = report_generator.create_analytical_report(
                        user=st.session_state.username,
                        ticker="STOCK-ANALYSIS",
                        recommendation=recommendation,
                        analytics_data=analytics_data,
                        historical_data=historical_data_for_ticker if 'historical_data_for_ticker' in locals() else None
                    )
                    
                    # Use a more visible approach for the download
                    st.success("✅ Stock Analysis Report successfully generated!")
                    
                    # Read the file for direct download button
                    with open(report_path, "rb") as f:
                        pdf_data = f.read()
                    
                    # Create a prominent download button
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                    )
                    
                    # Also display the HTML link as fallback
                    st.markdown("---")
                    st.markdown("If the button above doesn't work, use this alternative link:")
                    download_link = report_generator.get_download_link(report_path)
                    st.markdown(download_link, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error generating PDF report: {str(e)}")
            else:
                st.info("Generate analytics in the 'Stock Forecast > Analytics' tab or get an AI recommendation in the 'Get Advice' tab")
        
        elif report_type == "Forecast Report":
            # Check if we have forecast data to save
            forecast_data = st.session_state.get('forecast_data', None)
            lstm_forecast = st.session_state.get('lstm_forecast', None)
            
            if forecast_data or lstm_forecast:
                st.write("Your forecast report is ready to save!")
                
                # Choose which forecast to include
                if forecast_data and lstm_forecast:
                    forecast_choice = st.radio(
                        "Choose forecast to include:", 
                        ["Quick Forecast", "LSTM January 2025 Forecast", "Both"]
                    )
                    
                    if forecast_choice == "Quick Forecast":
                        selected_forecast = forecast_data
                    elif forecast_choice == "LSTM January 2025 Forecast":
                        selected_forecast = lstm_forecast
                    else:
                        # Combine both forecasts
                        selected_forecast = {
                            'ticker': forecast_data.get('ticker', ''),
                            'metrics': forecast_data.get('metrics', {}),
                            'forecast_df': forecast_data.get('forecast_df', None),
                            'lstm_forecast_df': lstm_forecast.get('forecast_df', None)
                        }
                else:
                    selected_forecast = forecast_data or lstm_forecast
                
                # Create PDF report with historical data for comparison
                historical_data = None
                if selected_forecast:
                    # Get historical data for the selected ticker
                    df_ticker = df[df['Ticker'] == selected_forecast['ticker']]
                    if not df_ticker.empty:
                        historical_data = df_ticker.sort_values('Date').tail(90)  # Last 90 days
                
                # Generate the PDF report
                try:
                    report_generator = PDFReportGenerator(output_dir="reports")
                    report_path = report_generator.create_analytical_report(
                        user=st.session_state.username,
                        ticker=selected_forecast.get('ticker', 'STOCK'),
                        forecast_data=selected_forecast,
                        historical_data=historical_data
                    )
                    
                    # Use a more visible approach for the download
                    st.success("✅ PDF Report successfully generated!")
                    
                    # Read the file for direct download button
                    with open(report_path, "rb") as f:
                        pdf_data = f.read()
                    
                    # Create a prominent download button
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                    )
                    
                    # Also display the HTML link as fallback
                    st.markdown("---")
                    st.markdown("If the button above doesn't work, use this alternative link:")
                    download_link = report_generator.get_download_link(report_path)
                    st.markdown(download_link, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error generating PDF report: {str(e)}")
            else:
                st.info("Generate a forecast in the 'Stock Forecast' tab first")
        
        else:  # CrewAI Analysis
            if st.session_state.crew_analysis_results:
                st.write("Your CrewAI analysis report is ready to save!")
                
                # Generate a comprehensive PDF for CrewAI analysis using our new module
                try:
                    # Initialize the PDF report generator
                    report_generator = PDFReportGenerator(output_dir="reports")
                    
                    results = st.session_state.crew_analysis_results
                    
                    # Extract analytics data from the CrewAI results
                    analytics_data = {}
                    if 'analysis' in results and 'metrics' in results['analysis']:
                        # Create company comparison data
                        metrics_data = results['analysis']['metrics']
                        company_data = []
                        
                        for ticker, metric in metrics_data.items():
                            company_data.append({
                                'ticker': ticker,
                                'highest_price': metric.get('highest_price', 'N/A'),
                                'lowest_price': metric.get('lowest_price', 'N/A'),
                                'annual_growth': metric.get('annual_growth_2020', 'N/A'),
                                'volatility': metric.get('volatility', 'N/A'),
                                'current_price': metric.get('current_price', 'N/A')
                            })
                        
                        analytics_data['company_comparison'] = pd.DataFrame(company_data)
                    
                    # Extract LLM insights from the CrewAI results
                    llm_insights = {}
                    if 'recommendations' in results:
                        # Add market summary if available
                        if 'market_summary' in results['recommendations']:
                            llm_insights['summary'] = results['recommendations']['market_summary']
                        
                        # Add ticker recommendations if available
                        if 'ticker_recommendations' in results['recommendations']:
                            recommendations_text = ""
                            for ticker, rec in results['recommendations']['ticker_recommendations'].items():
                                rec_type = rec.get('recommendation_type', 'UNKNOWN')
                                rec_text = rec.get('recommendation_text', 'No recommendation available')
                                recommendations_text += f"{ticker} - {rec_type}\n{rec_text}\n\n"
                            
                            llm_insights['recommendations'] = recommendations_text
                    
                    # Generate the CrewAI analysis report
                    report_path = report_generator.create_analytical_report(
                        user=st.session_state.username,
                        ticker="MULTI-STOCK",  # CrewAI analyzes multiple stocks
                        analytics_data=analytics_data,
                        llm_insights=llm_insights
                    )
                    
                    # Use a more visible approach for the download
                    st.success("✅ CrewAI Analysis Report successfully generated!")
                    
                    # Read the file for direct download button
                    with open(report_path, "rb") as f:
                        pdf_data = f.read()
                    
                    # Create a prominent download button
                    st.download_button(
                        label="📥 Download CrewAI Report",
                        data=pdf_data,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                    )
                    
                    # Also display the HTML link as fallback
                    st.markdown("---")
                    st.markdown("If the button above doesn't work, use this alternative link:")
                    download_link = report_generator.get_download_link(report_path)
                    st.markdown(download_link, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
            else:
                st.info("Run a CrewAI analysis in the 'CrewAI Analysis' tab first to generate a report")

# Run appropriate UI based on login state
if st.session_state.logged_in:
    main_app()
else:
    auth_ui()