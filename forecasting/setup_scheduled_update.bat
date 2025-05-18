@echo off
:: Setup scheduled updates for stock data
echo Setting up scheduled task for daily stock data updates...

:: Get the current directory path
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

:: Create the task 
schtasks /create /tn "MarketEyeAI_Stock_Update" /tr "python %PROJECT_DIR%\forecasting\data_updater.py --update" /sc DAILY /st 00:00 /f

:: Also create a separate task that runs backfilling until December 31, 2024
schtasks /create /tn "MarketEyeAI_Stock_Backfill" /tr "python %PROJECT_DIR%\forecasting\data_updater.py --backfill --end-date 2024-12-31" /sc MONTHLY /mo 1 /st 01:00 /f

echo.
echo Tasks created:
echo - MarketEyeAI_Stock_Update: Runs daily at 00:00 to update stock data
echo - MarketEyeAI_Stock_Backfill: Runs monthly to ensure backfilling until December 31, 2024
echo.
echo You can verify these tasks in the Windows Task Scheduler.
echo.

pause 