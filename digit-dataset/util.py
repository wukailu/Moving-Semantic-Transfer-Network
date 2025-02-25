import logging
import os.path
from tqdm import tqdm

import requests

logger = logging.getLogger(__name__)


def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)


def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        print("downloading...")
        for chunk in tqdm(response.iter_content(chunk_size=1024),
                          total=int(response.headers.get('content-length'))//1024+1):
            if chunk:
                f.write(chunk)
