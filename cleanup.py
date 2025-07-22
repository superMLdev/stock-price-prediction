#!/usr/bin/env python3
"""
cleanup.py - Script to clean up unnecessary files in the stock price prediction project

This script removes:
1. Generated visualization files (outside docs directory)
2. Trained model files (while keeping templates)
3. Log files
4. Unnecessary script files
5. Temporary and cache files

Usage:
    python cleanup.py [--keep-data]

Options:
    --keep-data    Keep the data directory with downloaded stock data
"""

import os
import shutil
import argparse
import glob

def confirm(message):
    """Ask for confirmation before proceeding"""
    response = input(f"{message} [y/N]: ").lower()
    return response in ('y', 'yes')

def remove_file(filepath):
    """Safely remove a file with confirmation"""
    if os.path.exists(filepath):
        print(f"Removing file: {filepath}")
        os.remove(filepath)

def remove_directory(dirpath):
    """Safely remove a directory with confirmation"""
    if os.path.exists(dirpath):
        print(f"Removing directory: {dirpath}")
        shutil.rmtree(dirpath)

def main():
    parser = argparse.ArgumentParser(description="Clean up unnecessary files in the project.")
    parser.add_argument('--keep-data', action='store_true', help='Keep the data directory with downloaded stock data')
    args = parser.parse_args()
    
    # Project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Confirm before proceeding
    if not confirm("This will remove visualization files, model files, logs, and other unnecessary files. Continue?"):
        print("Operation cancelled.")
        return
    
    # 1. Remove visualization files (except those in docs)
    print("\n--- Cleaning visualization files ---")
    # Remove PNG files in root directory
    for png_file in glob.glob(os.path.join(root_dir, "*.png")):
        remove_file(png_file)
    
    # Remove visualizations directory
    if os.path.exists(os.path.join(root_dir, "visualizations")):
        remove_directory(os.path.join(root_dir, "visualizations"))
    
    # 2. Clean models directory
    print("\n--- Cleaning model files ---")
    models_dir = os.path.join(root_dir, "models")
    if os.path.exists(models_dir):
        # Remove all model files except templates
        for file in os.listdir(models_dir):
            file_path = os.path.join(models_dir, file)
            # Keep basic template models but remove stock-specific ones
            if file.startswith(("AAPL_", "MSFT_", "GOOGL_", "NVDA_", "AMZN_")):
                remove_file(file_path)
    
    # 3. Remove log files
    print("\n--- Cleaning log files ---")
    for log_file in glob.glob(os.path.join(root_dir, "*.log")):
        remove_file(log_file)
    
    # 4. Clean data directory if not keeping it
    if not args.keep_data:
        print("\n--- Cleaning data directory ---")
        data_dir = os.path.join(root_dir, "data")
        if os.path.exists(data_dir):
            remove_directory(data_dir)
            # Recreate empty data directory
            os.makedirs(data_dir, exist_ok=True)
            print(f"Created empty data directory: {data_dir}")
    
    # 5. Remove predictions directory
    print("\n--- Cleaning predictions directory ---")
    predictions_dir = os.path.join(root_dir, "predictions")
    if os.path.exists(predictions_dir):
        remove_directory(predictions_dir)
        # Recreate empty predictions directory
        os.makedirs(predictions_dir, exist_ok=True)
        print(f"Created empty predictions directory: {predictions_dir}")
    
    # 6. Remove unnecessary script files
    print("\n--- Cleaning unnecessary script files ---")
    unnecessary_scripts = [
        "compare_models.py", 
        "compare_all_models.py",
        "load_and_predict_lstm.py"
    ]
    for script in unnecessary_scripts:
        script_path = os.path.join(root_dir, script)
        if os.path.exists(script_path):
            remove_file(script_path)
    
    # 7. Clean cache files
    print("\n--- Cleaning cache files ---")
    # Remove __pycache__ directories
    for pycache_dir in glob.glob(os.path.join(root_dir, "**/__pycache__"), recursive=True):
        remove_directory(pycache_dir)
    
    # Remove .pyc files
    for pyc_file in glob.glob(os.path.join(root_dir, "**/*.pyc"), recursive=True):
        remove_file(pyc_file)
    
    print("\nCleanup complete!")
    print("""
Recommended files for users:
- train_and_predict.py: Main script for training models and making predictions
- load_and_predict.py: Script for loading pre-trained models and making predictions
- algorithms/: Core model implementations
- requirement.txt: Dependencies for installation
""")

if __name__ == "__main__":
    main()
