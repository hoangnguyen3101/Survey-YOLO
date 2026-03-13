from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="/home/hoangnv/YOLO/runs", repo_id="hoangnguyen311111/version_YOLO_with_bdd10k", repo_type="model")
