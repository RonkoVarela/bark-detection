from huggingface_hub import HfApi

# Note that you may need to `huggingface-cli login` before executing this script

if __name__ == '__main__':
    api = HfApi()
    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    for split in ["train", "test", "validation"]:
        for label in ["yes", "no"]:
            api.upload_folder(
                folder_path=f"./records/{split}/{label}",
                repo_id="rmarcosg/bark-detection",
                path_in_repo=f"{split}/{label}/",
                repo_type="dataset",
            )
