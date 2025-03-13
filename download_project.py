
import os
import zipfile
from datetime import datetime

def zip_project():
    # Define zip filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"trading_bot_project_{timestamp}.zip"
    
    # List of files to exclude
    exclude = ['.git', '.config', '__pycache__', '.upm', '.cache', zip_filename]
    
    print(f"Creating zip file: {zip_filename}")
    
    # Create zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude]
            
            for file in files:
                # Skip the zip file itself
                if file == zip_filename:
                    continue
                    
                file_path = os.path.join(root, file)
                # Add file to zip
                print(f"Adding: {file_path}")
                zipf.write(file_path)
    
    print(f"\nZip file created: {zip_filename}")
    print("You can download this file from the Files panel in Replit")

if __name__ == "__main__":
    zip_project()
