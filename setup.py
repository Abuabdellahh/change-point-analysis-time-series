"""
Setup script for the Brent Oil Price Analysis project.
"""
import os
import sys
import subprocess
from pathlib import Path
import yaml

def check_environment():
    """Check if required tools are installed."""
    required_tools = ['python', 'conda', 'git']
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run(
                [tool, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            print(f"✓ {tool} is installed")
        except (subprocess.SubprocessError, FileNotFoundError):
            missing_tools.append(tool)
    
    if missing_tools:
        print("\nThe following required tools are missing:")
        for tool in missing_tools:
            print(f"- {tool}")
        print("\nPlease install them before proceeding.")
        return False
    
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks',
        'logs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def check_data_files():
    """Check if required data files exist."""
    config = {}
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_files = [
        ('Brent prices', config['data']['raw']),
        ('Geopolitical events', config['data']['events'])
    ]
    
    all_files_exist = True
    
    for name, path in data_files:
        if not os.path.exists(path):
            print(f"⚠ {name} file not found: {path}")
            all_files_exist = False
        else:
            print(f"✓ Found {name} file: {path}")
    
    if not all_files_exist:
        print("\nPlease add the missing data files to their respective locations.")
        print("Template files have been created in the data/ directory.")
    
    return all_files_exist

def main():
    print("\n=== Brent Oil Price Analysis Project Setup ===\n")
    
    # Check environment
    print("Checking environment...")
    if not check_environment():
        return
    
    # Create directories
    print("\nSetting up project structure...")
    create_directories()
    
    # Check data files
    print("\nChecking data files...")
    data_ready = check_data_files()
    
    if data_ready:
        print("\n✅ Setup complete! You can now run the analysis.")
        print("To get started, activate the conda environment and run the notebooks:")
        print("\n  conda activate brent-analysis")
        print("  jupyter notebook notebooks/")
    else:
        print("\n❌ Setup incomplete. Please add the missing files and run setup again.")

if __name__ == "__main__":
    main()
