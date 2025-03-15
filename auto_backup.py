import os
import time

while True:
    os.system("git add .")
    os.system('git commit -m "Auto Backup: $(date)"')
    os.system("git push origin main")

    print("✅ Backup completed! Waiting for next cycle...")
    time.sleep(1800)  # Wait 30 minutes before the next backup
