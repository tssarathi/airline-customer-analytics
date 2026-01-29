import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
RAW_FOLDER = os.getenv("RAW_PREFIX")
PROCESSED_FOLDER = os.getenv("PROCESSED_PREFIX")
GLUE_DATABASE = os.getenv("GLUE_DB")
ATHENA_RESULTS_FOLDER = os.getenv("ATHENA_RESULTS_PREFIX")
CURATED_FOLDER = os.getenv("CURATED_PREFIX")
