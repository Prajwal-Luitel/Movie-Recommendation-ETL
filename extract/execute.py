import time
import os
import sys
import requests
from zipfile import ZipFile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.utility import setup_logging, format_time


def download_zip_file(logger, url, output_dir):
    response = requests.get(url,stream=True)
    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Downloading start")
    if response.status_code == 200:
        filename = os.path.join(output_dir, "downloaded.zip")
        with open(filename,"wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded zip file : {filename}")
        return filename
    else:
        logger.error("Unsuccessfull")
        raise Exception(f"Failed to download file. Status coe {response.status_code}")
    
def extract_zip_file(logger,zip_filename, output_dir):
    logger.info("Extracting the zip file")
    with ZipFile(zip_filename, "r") as zip_file:
        zip_file.extractall(output_dir)
    
    logger.info(f"Extracted files written to : {output_dir}")
    logger.info("Removing the zip file")
    os.remove(zip_filename)


if __name__ == "__main__":

    logger = setup_logging("extract.log")

    if len(sys.argv) < 2:
        logger.error("Extraction path is required")
        logger.error("Exame Usage:")
        logger.error("python3 execute.py /home/prajwal/Movie-Data/Extraction")
    else:
        try:
            logger.info("Starting Extraction Engine...")
            EXTRACT_PATH = sys.argv[1]
            KAGGLE_URL = "https://storage.googleapis.com/kaggle-data-sets/5499347/9111436/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250810%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250810T104427Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=14dc7e5b3af803d31b145129f678da6b90d7584375941b1bd91c1f424a0deac77f42fcf6c6da5e0200cba556b64d81195e5395c8941171d7561d899ab6be963c31939a231921a7094aa25df166f35050bfc8bd758346086aa0bcf89acff9824954adcb817481aed09648d208ed30f553ac39c297ccf0c1c807385bc109d3ffe40fea95ec2976c3e2529248bfb9bb8f9cfc85ae8c6279020e4f9ce824080d3079e32198b146d89b6eec17af655b5c2e3566d34b32c88d83484c893dd00c2d122c4d24ff4dbb6f5028f3b89248db2ce07a2878da72d8fb49df6bcd4ea1ee82b54b1ea20cc1c44b862d999d2d7c202e77284d0c8b17aacaa97b259c4a6043f973ee"
            
            start = time.time()
            zip_filename = download_zip_file(logger, KAGGLE_URL,EXTRACT_PATH)
            extract_zip_file(logger, zip_filename, EXTRACT_PATH)
            end = time.time()
            logger.info("Extraction Sucessfully Complete!!!")
            logger.info(f"Total time taken {format_time(end-start)}")

        except Exception as e:
            logger.error(f"Error: {e}")