@echo off
setlocal

set SCRIPT_PATH=C:\Users\vasil\Downloads\football\webScrapper\getStats.py
set PYTHON_EXECUTABLE=C:\Users\vasil\AppData\Local\Programs\Python\Python311\python.exe
set RUNNING_FROM_BATCH=1

:loop
rem Run the Python script
%PYTHON_EXECUTABLE% %SCRIPT_PATH%

rem Check the error level and act accordingly
if %errorlevel% NEQ 0 (
    echo Script encountered an error. Restarting...
    goto loop
)

echo Script completed successfully.
