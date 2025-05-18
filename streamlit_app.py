# streamlit_app.py

import streamlit as st
import pandas as pd
from fpdf import FPDF
import base64
import google.generativeai as genai
import sqlite3
import os
import requests
import json
from datetime import datetime

# Import our forecasting API
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forecasting.forecast_api import ForecastAPI

# Import CrewAI components
from agents.stock_analysis_crew import StockAnalysisCrew

# Configure page
st.set_page_config(page_title="Market Eye AI", layout="wide")

# Database connection
def get_db_connection():
    conn = sqlite3.connect('market_eye.db')
    conn.row_factory = sqlite3.Row
    return conn

# Authentication logic
def authenticate_user(username, password):
    try:
        # Create a dummy email by appending @example.com to username
        # This is needed because the backend API expects an email field
        dummy_email = f"{username}@example.com"
        
        response = requests.post(
            "http://127.0.0.1:8000/login",
            json={"username": username, "password": password, "email": dummy_email}
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"Login failed with status code: {response.status_code}")
            if response.status_code == 422:
                print("Validation error in request format")
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

# Configure Gemini AI 
genai.configure(api_key="AIzaSyCcB3tENdnFNKBIvciETMF196ldlUBmnyk")
model = genai.GenerativeModel("gemini-1.5-flash")

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
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
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

# Function to create PDF reports
def generate_pdf_report(recommendation, user):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add content to PDF
    pdf.cell(200, 10, txt="Market Eye AI - Stock Analysis Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Generated for: {user}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.multi_cell(0, 10, txt="\nAI Recommendation:\n" + recommendation)
    
    # Save PDF and create download link
    pdf.output("report.pdf")
    with open("report.pdf", "rb") as f:
        pdf_data = f.read()
    b64 = base64.b64encode(pdf_data).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF Report</a>'

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
                            
                            response = model.generate_content(prompt)
                            st.subheader("AI Recommendation")
                            st.write(response.text)
                            
                            # Save for PDF report
                            st.session_state.recommendation = response.text
                            
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
        
        if not available_tickers:
            st.warning("No forecasting models available. Please train models first.")
            
            # Enhanced dataset option
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
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
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
                    updated_df['Date'] = pd.to_datetime(updated_df['Date'], errors='coerce')
                    
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
        report_type = st.radio("Report Type", ["Simple Recommendation", "CrewAI Analysis"], horizontal=True)
        
        if report_type == "Simple Recommendation":
            # Check if we have a recommendation to save
            if "recommendation" in st.session_state:
                st.write("Your recommendation report is ready to save!")
                
                # Create a more comprehensive report
                pdf_html = generate_pdf_report(st.session_state.recommendation, st.session_state.username)
                st.markdown(pdf_html, unsafe_allow_html=True)
            else:
                st.info("First get an AI recommendation in the 'Get Advice' tab or generate a forecast in the 'Stock Forecast' tab")
        
        else:  # CrewAI Analysis
            if st.session_state.crew_analysis_results:
                st.write("Your CrewAI analysis report is ready to save!")
                
                # Generate a more comprehensive PDF for CrewAI analysis
                try:
                    # Create PDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Add content to PDF
                    pdf.cell(200, 10, txt="Market Eye AI - CrewAI Stock Analysis Report", ln=True, align='C')
                    pdf.cell(200, 10, txt=f"Generated for: {st.session_state.username}", ln=True, align='C')
                    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
                    
                    results = st.session_state.crew_analysis_results
                    
                    # Add market summary
                    if 'recommendations' in results and 'market_summary' in results['recommendations']:
                        pdf.ln(10)
                        pdf.set_font("Arial", 'B', size=12)
                        pdf.cell(0, 10, txt="Market Summary", ln=True)
                        pdf.set_font("Arial", size=10)
                        
                        # Split the market summary into smaller chunks to fit in PDF
                        market_summary = results['recommendations']['market_summary']
                        chunks = [market_summary[i:i+100] for i in range(0, len(market_summary), 100)]
                        for chunk in chunks:
                            pdf.multi_cell(0, 5, txt=chunk)
                    
                    # Add recommendations for each ticker
                    if 'recommendations' in results and 'ticker_recommendations' in results['recommendations']:
                        recommendations = results['recommendations']['ticker_recommendations']
                        
                        pdf.ln(10)
                        pdf.set_font("Arial", 'B', size=12)
                        pdf.cell(0, 10, txt="Stock Recommendations", ln=True)
                        
                        for ticker, rec in recommendations.items():
                            pdf.set_font("Arial", 'B', size=10)
                            rec_type = rec.get('recommendation_type', 'UNKNOWN')
                            pdf.cell(0, 10, txt=f"{ticker} - {rec_type}", ln=True)
                            
                            pdf.set_font("Arial", size=10)
                            rec_text = rec.get('recommendation_text', 'No recommendation available')
                            
                            # Split the recommendation into smaller chunks
                            chunks = [rec_text[i:i+100] for i in range(0, len(rec_text), 100)]
                            for chunk in chunks:
                                pdf.multi_cell(0, 5, txt=chunk)
                            
                            pdf.ln(5)
                    
                    # Save PDF and create download link
                    pdf.output("crewai_report.pdf")
                    with open("crewai_report.pdf", "rb") as f:
                        pdf_data = f.read()
                    b64 = base64.b64encode(pdf_data).decode()
                    
                    st.markdown(
                        f'<a href="data:application/pdf;base64,{b64}" download="crewai_report.pdf">Download CrewAI Analysis PDF Report</a>', 
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
            else:
                st.info("Run a CrewAI analysis in the 'CrewAI Analysis' tab first to generate a report")

# Run appropriate UI based on login state
if st.session_state.logged_in:
    main_app()
else:
    auth_ui()