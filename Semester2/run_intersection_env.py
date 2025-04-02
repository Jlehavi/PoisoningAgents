#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil

def ensure_data_files_exist():
    """Ensure that necessary data files exist"""
    # Check if intersection_data.txt exists
    if not os.path.exists("intersection_data.txt"):
        # Copy from Semester2 directory if it exists there
        if os.path.exists("Semester2/intersection_data.txt"):
            shutil.copy("Semester2/intersection_data.txt", "intersection_data.txt")
            print("Copied intersection_data.txt from Semester2 directory")
        else:
            print("WARNING: intersection_data.txt not found! The RAG system may not work properly.")

def run_intersection_env():
    """Run the IntersectionEnv.py script using the existing Python environment"""
    print("Running IntersectionEnv.py...")
    subprocess.run([sys.executable, "IntersectionEnv.py"])

if __name__ == "__main__":
    ensure_data_files_exist()
    run_intersection_env() 