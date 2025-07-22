#!/bin/bash

# Script to build and serve the Jekyll documentation

# Navigate to the docs directory
cd "$(dirname "$0")/docs" || exit

# Check if Ruby is installed
if ! command -v ruby &> /dev/null; then
    echo "Ruby is not installed. Please install Ruby before continuing."
    exit 1
fi

# Check if Bundler is installed
if ! command -v bundle &> /dev/null; then
    echo "Bundler is not installed. Installing now..."
    gem install bundler
fi

# Install dependencies if needed
if [ ! -f "Gemfile.lock" ]; then
    echo "Installing dependencies..."
    bundle install
fi

# Build and serve the site
echo "Starting Jekyll server..."
echo "Documentation will be available at: http://localhost:4000"
echo "Press Ctrl+C to stop the server"
bundle exec jekyll serve
