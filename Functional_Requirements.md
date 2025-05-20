# Functional Requirements

## Table of Contents

- [1. Database & User Management](#1--database--user-management)
- [2. CrewAI Agents](#2--crewai-agents)
- [3. Dataset Integration](#3--dataset-integration)
- [4. Analysis & Forecasting](#4--analysis--forecasting)
- [5. Streamlit Front-End](#5--streamlit-front-end)
- [6. PDF Report Generation](#6--pdf-report-generation)
- [7. LLM Integration](#7--llm-integration)
- [8. GitHub & Version Control](#8--github--version-control)

---

## 1- Database & User Management

### Steps:

- Implement user authentication: sign-up and login pages.
- Securely hash and store passwords.
- Define and create two tables:
  - users(user_id, username, password_hashed, registration_date)
  - activity_logs(user_id, action, timestamp)
- Log every user action (registration, login, app usage) into activity_logs.

### Deliverable:

> Working user authentication module with users and activity_logs tables fully
> implemented and logging enabled.

---

## 2- CrewAI Agents

### Steps:

- **Agent 1 – Data Collector:**
  - Load "World Stock Prices" data from Kaggle for specified tickers.
  - Clean and prepare Pandas DataFrames.
- **Agent 2 – Data Processor:**
  - Compute high, low, and 2020 growth% for each ticker.
  - Compare sectors: Tech vs. Finance vs. Sportswear.
  - Train and use an LSTM/MLP model on historical "Close" prices.
  - Predict January 2025 prices and (once available) calculate MSE & RMSE.
- **Agent 3 – LLM Recommendation Generator:**
  - Build prompts dynamically (company + analysis + forecast + errors).
  - Call the Gemini API to generate market-trend summaries and Buy/Hold/Sell recommendations.

### Deliverable:

> Three deployed CrewAI agents: Data Collector, Data Processor, and LLM
> Recommendation Generator, all tested end-to-end.

---

## 3- Dataset Integration

### Steps:

- Ingest the Kaggle "World Stock Prices – Daily Updating" dataset.
- Automate daily updates through December 2024.

### Deliverable:

> A scheduled pipeline that fetches and updates the stock-price dataset daily up to Dec
> 31, 2024.

---

## 4- Analysis & Forecasting

### Steps:

- Compute analytics: highest price, lowest price, and annual growth for each ticker.
- Generate cross-company and cross-sector comparisons.
- Train forecasting model on "Close" prices.
- Produce January 2025 forecasts.
- Evaluate forecasts by calculating MSE and RMSE against real Jan 2025 data.

### Deliverable:

> Analytical reports, comparison charts, forecast outputs, and error metrics (MSE, RMSE).

---

## 5- Streamlit Front-End

### Steps:

- Build authentication UI: sign-up and sign-in forms.
- Add a ticker selector (single or multi-select dropdown).
- Create dashboard pages for: analytics tables, forecast results, and real vs. forecast charts.
- Implement a PDF-report download button.
- Display LLM-generated recommendations.

### Deliverable:

> Deployed Streamlit app with all pages, selector, charts, PDF export, and
> recommendation panel.

---

## 6- PDF Report Generation

### Steps:

- Use ReportLab or FPDF to assemble:
  - Analytical tables and figures.
  - Forecast vs. real-data comparisons.
  - Error metrics (MSE, RMSE).
  - LLM-generated text.

### Deliverable:

> A PDF-generation module producing comprehensive, formatted stock analysis reports.

---

## 7- LLM Integration

### Steps:

- Integrate Google Gemini API.
- Dynamically assemble prompts from agent outputs.
- Ensure responses are fully AI-generated (no static templates).

### Deliverable:

> Functional Gemini API integration that returns contextual market insights and
> recommendations.

---

## 8- GitHub & Version Control

### Steps:

- Organize repository with clear folder structure (agents, models, frontend, backend).
- Enforce frequent commits with descriptive messages.
- Include a .gitignore and contribute guidelines.

### Deliverable:

> A public GitHub repo with logical structure, documented commit history, and contribution
> instructions.
