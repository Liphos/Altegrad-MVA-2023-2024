@echo off
REM Set the current directory to the directory of the batch file
cd /d "%~dp0"
call ".venv/Scripts/activate"
echo %1
python create_submission.py %1
pause