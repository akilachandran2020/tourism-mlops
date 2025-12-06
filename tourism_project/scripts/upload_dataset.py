from huggingface_hub import HfApi, upload_folder
import os

api = HfApi()

repo_id = "AkilaChandran2025/tourism"  # change if needed

upload_folder(
    folder_path="./test.csv",  # where train.csv/test.csv are
    repo_id=repo_id,
    repo_type="dataset"
)

print("Dataset uploaded successfully!")
