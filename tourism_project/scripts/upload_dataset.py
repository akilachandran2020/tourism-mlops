import os
from huggingface_hub import HfApi, upload_folder

repo_id = "AkilaChandran2025/tourism_dataset"   # change if needed

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Copy dataset files into dataset/
if os.path.exists("tourism_project/data/tourism.csv"):
    os.system("cp tourism_project/data/tourism.csv dataset/")
if os.path.exists("tourism_project/data/train.csv"):
    os.system("cp tourism_project/data/train.csv dataset/")
if os.path.exists("tourism_project/data/test.csv"):
    os.system("cp tourism_project/data/test.csv dataset/")

# Upload dataset folder
upload_folder(
    folder_path="dataset",
    repo_id=repo_id,
    repo_type="dataset"
)

print("Dataset uploaded successfully!")
