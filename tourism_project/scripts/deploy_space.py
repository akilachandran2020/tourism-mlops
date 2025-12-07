import os
from huggingface_hub import HfApi

SPACE_ID = "AkilaChandran2025/tourism"

def main():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    api = HfApi(token=hf_token)

    # Only upload the files needed for the Space UI
    api.upload_folder(
        folder_path=".",                  # repo root in GitHub Actions
        repo_id=SPACE_ID,
        repo_type="space",
        allow_patterns=[
            "src/**",                    # your Streamlit app & related files
        ],
        ignore_patterns=[
            "README.md",
            "requirements.txt",
            ".git/**",
            ".github/**",
            "tourism_project/**",
            "*.ipynb",
        ],
        commit_message="Update Space UI from GitHub Actions",
    )

    print("Frontend deployed to Space")

if __name__ == "__main__":
    main()

print("Frontend deployed to Space!")
