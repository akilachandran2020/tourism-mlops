from huggingface_hub import upload_folder

upload_folder(
    folder_path=".",
    repo_id="AkilaChandran2025/tourism",   # your HF Space ID
    repo_type="space"
)

print("Frontend deployed to Space!")
