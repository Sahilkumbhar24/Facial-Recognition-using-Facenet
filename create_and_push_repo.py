#!/usr/bin/env python3
"""
Create a new GitHub repository and push project code automatically.
This script uses the GitHub API with your personal access token.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def main():
    # Configuration
    repo_name = "Facial-Recognition-FaceNet"
    repo_description = "Facial Recognition using FaceNet - Deep learning powered face recognition system"
    token = input("Enter your GitHub Personal Access Token: ").strip()
    username = input("Enter your GitHub username: ").strip()
    
    if not token or not username:
        print("❌ Token and username are required!")
        sys.exit(1)
    
    # API request to create repository
    print(f"\n📝 Creating repository '{repo_name}' on GitHub...")
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": repo_name,
        "description": repo_description,
        "private": False,
        "auto_init": False
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        repo_data = response.json()
        repo_url = repo_data["clone_url"]
        print(f"✅ Repository created successfully!")
        print(f"📍 URL: {repo_url}")
    else:
        print(f"❌ Failed to create repository: {response.json()}")
        sys.exit(1)
    
    # Navigate to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Add remote and push
    print(f"\n🚀 Pushing code to GitHub...")
    try:
        subprocess.run(["git", "remote", "remove", "origin"], capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        subprocess.run(["git", "push", "-u", "origin", "master"], check=True)
        print("✅ Code pushed successfully!")
        print(f"🎉 Your repository is ready at: https://github.com/{username}/{repo_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during push: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
