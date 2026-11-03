@echo off
cd /d "%~dp0"
python nba_feed.py --refresh
pause
