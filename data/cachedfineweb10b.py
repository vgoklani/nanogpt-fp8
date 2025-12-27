import logging
import os
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


datasets_directory = os.path.join(os.environ["HUGGINGFACE_HUB_DIRECTORY"], "datasets")


def download_dataset(
    repo_id: str, file_id: Optional[str] = None, max_workers: Optional[int] = 32
):
    if file_id is None:
        file_id = repo_id.replace(".", "__").replace("-", "_")

    local_dir = os.path.join(datasets_directory, file_id)
    log.info(f"downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=None,
        cache_dir=None,
        max_workers=max_workers,
        # ignore_patterns=["*.bin"],
        local_dir=local_dir,
        token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    )
    log.info(f"saved {repo_id} -> {local_dir}")


def main(repo_id: Optional[str] = "kjj0/finewebedu10B-gpt2"):
    download_dataset(repo_id=repo_id)


if __name__ == "__main__":
    try:
        __IPYTHON__
        print("\nrunning via ipython -> not running continously")
    except NameError:
        main()
