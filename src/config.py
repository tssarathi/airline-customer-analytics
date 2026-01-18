import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-4")
S3_BUCKET = os.getenv("S3_BUCKET", "airline-customer-analytics")
GLUE_DATABASE = os.getenv("GLUE_DB", "airline_analytics")