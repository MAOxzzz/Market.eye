# Market Eye AI - Executable Version

This is the standalone executable version of Market Eye AI, a powerful AI-powered stock analysis platform with LSTM forecasting, AI investment recommendations, and interactive visualization tools.

## How to Run the Application

### Windows Users

1. Navigate to the `Market Eye AI` folder
2. Double-click the `Market Eye AI.exe` file
3. The application will start both the backend and frontend servers automatically
4. A web browser should open to the application interface at http://localhost:8501

If the browser doesn't open automatically, manually navigate to:

- Frontend interface: http://localhost:8501
- Backend API docs: http://localhost:8000/docs

### Mac/Linux Users

1. Navigate to the `Market Eye AI` folder
2. Make the executable file executable (if needed): `chmod +x "Market Eye AI"`
3. Run the executable: `./Market\ Eye\ AI`
4. The application will start both the backend and frontend servers automatically
5. A web browser should open to the application interface at http://localhost:8501

## First-time Login

When you first start the application, you'll need to create an account:

1. Click on "Sign Up" on the login screen
2. Enter a username, email, and password
3. Click "Sign Up" to create your account
4. Log in with your new credentials

## Using the Application

The Market Eye AI platform includes several features:

1. **View Stocks**: Visualize historical stock data for different stocks
2. **Get Advice**: Receive AI-generated investment recommendations
3. **Stock Forecast**: View price predictions for up to 90 days
4. **CrewAI Analysis**: Get comprehensive AI-powered investment analysis
5. **Dataset Management**: Update and backfill stock data
6. **Generate Reports**: Create downloadable PDF reports

## Troubleshooting

### Application Won't Start

If the application fails to start:

1. Make sure no other applications are using ports 8000 or 8501
2. Try running the executable as administrator (Windows) or with sudo (Mac/Linux)
3. Check that your firewall isn't blocking the application

### Data Not Loading

If stock data isn't loading properly:

1. Make sure the `data/stock_data` folder exists in the application directory
2. Use the Dataset Management tab to upload a new dataset

### Connectivity Issues

If you're having trouble connecting to the application:

1. Make sure you're using http://localhost:8501 (not https)
2. Try a different browser
3. Restart the application

## Support and Feedback

If you encounter issues or have suggestions for improvement, please contact the development team at [your-email@example.com].

## Shutting Down

To properly shut down the application:

1. Press Ctrl+C in the console window, or
2. Close the console window where the application is running

This will ensure both the frontend and backend servers are properly terminated.
