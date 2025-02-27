#!/usr/bin/env python3
"""
Options Scanner - Setup Script
-----------------------------
This script helps set up the Options Scanner application by:
1. Installing required dependencies
2. Configuring the Schwab API credentials
3. Setting up the database
4. Testing the API connection
5. Creating scheduled tasks (optional)

Usage:
    python setup.py

"""

import os
import sys
import subprocess
import platform
import getpass

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def install_dependencies():
    """Install required Python packages"""
    print_header("Installing Dependencies")
    
    required_packages = [
        "streamlit", 
        "pandas", 
        "numpy", 
        "requests"
    ]
    
    for package in required_packages:
        print(f"Installing {package}...", end="")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_success("Done")
        except subprocess.CalledProcessError:
            print_error("Failed")
            print(f"Please install {package} manually: pip install {package}")
    
    print_success("All dependencies installed")

def configure_api_credentials():
    """Configure Schwab API credentials"""
    print_header("Configuring Schwab API Credentials")
    
    # Path to the scheduled fetch script
    script_path = "scheduled_fetch.py"
    
    if not os.path.exists(script_path):
        print_error(f"Cannot find {script_path}. Make sure you're in the correct directory.")
        return False
    
    # Get API credentials from user
    print_info("Enter your Schwab API credentials:")
    api_key = input("API Key: ").strip()
    api_secret = input("API Secret: ").strip()
    
    if not api_key or not api_secret:
        print_warning("Empty credentials provided. Using mock data for now.")
        use_mock = True
    else:
        use_mock = input("Use mock data instead of real API? (y/n, default: n): ").strip().lower() == 'y'
    
    # Read the current script
    with open(script_path, 'r') as file:
        lines = file.readlines()
    
    # Update the credentials and mock data flag
    with open(script_path, 'w') as file:
        for line in lines:
            if "API_KEY = " in line:
                file.write(f"API_KEY = \"{api_key}\"  # Replace with your actual API key\n")
            elif "API_SECRET = " in line:
                file.write(f"API_SECRET = \"{api_secret}\"  # Replace with your actual API secret\n")
            elif "USE_MOCK_DATA = " in line:
                file.write(f"USE_MOCK_DATA = {str(use_mock)}\n")
            else:
                file.write(line)
    
    print_success("API credentials configured")
    return True

def setup_database():
    """Set up the SQLite database"""
    print_header("Setting Up Database")
    
    # First check if streamlit app exists
    app_path = "app.py"
    if not os.path.exists(app_path):
        print_error(f"Cannot find {app_path}. Make sure you're in the correct directory.")
        return False
    
    # Execute the setup_database function from the main app
    try:
        print_info("Creating database schema...")
        # Create a temporary Python script to set up the database
        with open("temp_db_setup.py", "w") as f:
            f.write("""
import sqlite3

# Create database and tables
conn = sqlite3.connect('options_data.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS options_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    price REAL,
    exp_date TEXT,
    strike REAL,
    option_type TEXT,
    bid REAL,
    ask REAL,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility REAL,
    delta REAL,
    timestamp DATETIME
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS data_metadata (
    id INTEGER PRIMARY KEY,
    last_updated DATETIME,
    source TEXT
)
''')

conn.commit()
conn.close()
print("Database setup complete!")
            """)
        
        # Run the temporary script
        subprocess.check_call([sys.executable, "temp_db_setup.py"])
        os.remove("temp_db_setup.py")
        
        print_success("Database setup complete")
        return True
    except Exception as e:
        print_error(f"Error setting up database: {str(e)}")
        return False

def test_api_connection():
    """Test the Schwab API connection"""
    print_header("Testing API Connection")
    
    print_info("Running a test data fetch...")
    try:
        # Run the scheduled fetch script with a timeout
        result = subprocess.run(
            [sys.executable, "scheduled_fetch.py"],
            capture_output=True, 
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            print_success("API test successful!")
            print_info("Check the database to verify data was fetched.")
            return True
        else:
            print_error("API test failed")
            print_error(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print_error("API test timed out after 60 seconds")
        return False
    except Exception as e:
        print_error(f"Error testing API: {str(e)}")
        return False

def setup_scheduler():
    """Set up scheduled tasks based on the operating system"""
    print_header("Setting Up Scheduler")
    
    system = platform.system()
    
    if system == "Linux" or system == "Darwin":  # Linux or Mac
        print_info("For Linux/Mac systems, you'll need to set up a cron job.")
        print_info("Example cron job to run at 9:31 AM ET on weekdays:")
        script_path = os.path.abspath("scheduled_fetch.py")
        print(f"31 9 * * 1-5 {sys.executable} {script_path}")
        
        setup_cron = input("Would you like to set up this cron job now? (y/n): ").strip().lower() == 'y'
        
        if setup_cron:
            try:
                # Create a temporary file with the cron job
                with open("temp_cron", "w") as f:
                    f.write(f"31 9 * * 1-5 {sys.executable} {script_path}\n")
                
                # Add the cron job
                subprocess.run(["crontab", "-l"], stdout=open("existing_cron", "w"), stderr=subprocess.DEVNULL)
                subprocess.run("cat existing_cron temp_cron > new_cron", shell=True)
                subprocess.run(["crontab", "new_cron"])
                
                # Clean up temporary files
                for file in ["temp_cron", "existing_cron", "new_cron"]:
                    if os.path.exists(file):
                        os.remove(file)
                
                print_success("Cron job set up successfully")
            except Exception as e:
                print_error(f"Error setting up cron job: {str(e)}")
                print_info("Please set up the cron job manually using the example above.")
    
    elif system == "Windows":
        print_info("For Windows systems, you'll need to set up a Task Scheduler task.")
        print_info("Follow these steps:")
        print("1. Open Task Scheduler")
        print("2. Create a new Basic Task")
        print("3. Set it to run daily at 9:31 AM")
        print("4. Action: Start a program")
        print(f"5. Program/script: {sys.executable}")
        print(f"6. Add arguments: {os.path.abspath('scheduled_fetch.py')}")
        
        print_info("Would you like to open Task Scheduler now?")
        open_scheduler = input("(y/n): ").strip().lower() == 'y'
        
        if open_scheduler:
            try:
                subprocess.Popen(["taskschd.msc"])
                print_success("Task Scheduler opened")
            except Exception as e:
                print_error(f"Error opening Task Scheduler: {str(e)}")
    
    else:
        print_warning(f"Unsupported operating system: {system}")
        print_info("You'll need to set up a scheduled task manually to run scheduled_fetch.py daily.")

def run_streamlit_app():
    """Run the Streamlit app"""
    print_header("Running Streamlit App")
    
    app_path = "app.py"
    if not os.path.exists(app_path):
        print_error(f"Cannot find {app_path}. Make sure you're in the correct directory.")
        return False
    
    print_info("Starting Streamlit app...")
    
    try:
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path])
        print_success("Streamlit app started!")
        print_info("Open your browser to view the app (usually at http://localhost:8501)")
        return True
    except Exception as e:
        print_error(f"Error starting Streamlit app: {str(e)}")
        print_info("Run the app manually with: streamlit run app.py")
        return False

def main():
    """Main setup function"""
    print_header("Options Scanner Setup")
    
    print(f"{Colors.BOLD}This script will help you set up the Options Scanner application.{Colors.ENDC}")
    print("It will install dependencies, configure API credentials, set up the database,")
    print("test the API connection, and optionally set up scheduled tasks.")
    
    proceed = input("\nDo you want to continue? (y/n): ").strip().lower() == 'y'
    
    if not proceed:
        print_info("Setup cancelled.")
        return
    
    # Steps to perform
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Configuring API credentials", configure_api_credentials),
        ("Setting up database", setup_database),
        ("Testing API connection", test_api_connection),
        ("Setting up scheduler", setup_scheduler),
        ("Running Streamlit app", run_streamlit_app)
    ]
    
    # Execute each step
    success = True
    for step_name, step_func in steps:
        print_header(step_name)
        
        # Ask if user wants to perform this step
        do_step = input(f"Do you want to {step_name.lower()}? (y/n, default: y): ").strip().lower() != 'n'
        
        if do_step:
            result = step_func()
            if not result and step_name != "Setting up scheduler" and step_name != "Running Streamlit app":
                # Critical steps failed
                success = False
                print_error(f"{step_name} failed. Setup cannot continue.")
                break
        else:
            print_info(f"Skipping {step_name.lower()}")
    
    if success:
        print_header("Setup Complete")
        print_success("The Options Scanner has been set up successfully!")
        print_info("You can now use the app to scan for options opportunities.")
    else:
        print_header("Setup Incomplete")
        print_warning("The Options Scanner setup is incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()
