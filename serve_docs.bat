@echo off
REM Script to build and serve the Jekyll documentation on Windows

REM Navigate to the docs directory
cd "%~dp0\docs" || exit /b

REM Check if Ruby is installed
where ruby >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Ruby is not installed. Please install Ruby before continuing.
    exit /b 1
)

REM Check if Bundler is installed
where bundle >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Bundler is not installed. Installing now...
    gem install bundler
)

REM Install dependencies if needed
if not exist Gemfile.lock (
    echo Installing dependencies...
    call bundle install
)

REM Build and serve the site
echo Starting Jekyll server...
echo Documentation will be available at: http://localhost:4000
echo Press Ctrl+C to stop the server
call bundle exec jekyll serve
