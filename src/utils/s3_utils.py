import boto3
from botocore.exceptions import ClientError


def s3_object_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    path = s3_uri.removeprefix("s3://")
    bucket, _, key = path.partition("/")
    return bucket, key


def upload_file_to_s3(local_path: str, output_s3_uri: str) -> None:
    s3 = boto3.client("s3")

    bucket, key = parse_s3_uri(output_s3_uri)
    if s3_object_exists(bucket, key):
        print(f"File already exists in S3 at s3://{bucket}/{key}")
        return

    s3.upload_file(local_path, bucket, key)
    print(f"File uploaded to s3://{bucket}/{key}")
