import os
import pickle

import braceexpand
from tqdm import tqdm

from .build_index import build_lid_index


def build_urls(
    langs: list[str] = [],
    index_path: str = "./data/lid_index.pkl",
    hf_token: str = None,
) -> list[str]:
    """
    Build a list of WebDataset URLs for the given languages.
    If langs is empty, all languages are included.
    Args:
        langs (list): List of language codes to include. If empty, all languages are included.
        index_path (str): Path to the language ID index folder.
    Returns:
        list[str]: List of WebDataset URLs.
    """

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set")
    token = f"Authorization:Bearer {hf_token}"
    if len(langs) > 0:
        if not os.path.exists(index_path):
            build_lid_index(index_path, hf_token=hf_token)

        with open(index_path, "rb") as f:
            lid_index = pickle.load(f)

        all_relevant_tar_numbers = set()
        for (tar_number, _), lang in tqdm(lid_index.items()):
            if lang in langs:
                all_relevant_tar_numbers.add(tar_number)
        all_relevant_tar_numbers = list(all_relevant_tar_numbers)
        urls = []
        for tar_number in all_relevant_tar_numbers:
            if int(tar_number) <= 5000:
                urls.append(
                    f"https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio/{tar_number}.tar?download=True"
                )
            else:
                urls.append(
                    f"https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio2/{tar_number}.tar?download=True"
                )
        urls = [f"pipe:curl -s -L {url} -H {token}" for url in urls]
        return urls

    else:
        # Choose the number of tars to download
        url = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/audio/{000001..000004}.tar?download=True"
        urls = list(braceexpand.braceexpand(url))
        urls = [f"pipe:curl -s -L {url} -H {token}" for url in urls]
        return urls
