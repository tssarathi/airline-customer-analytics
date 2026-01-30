import boto3
from sagemaker.core.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import ProcessingS3Input, ProcessingS3Output
from sagemaker.core.image_uris import retrieve

from src.config.config import (
    AWS_REGION,
    CURATED_FOLDER,
    PROCESSING_INSTANCE_TYPE,
    S3_BUCKET,
    ROLE_ARN,
)


def run_processing_job():
    input_s3 = (
        f"s3://{S3_BUCKET}/{CURATED_FOLDER}/customer_features/customer_features.parquet"
    )
    output_s3 = f"s3://{S3_BUCKET}/{CURATED_FOLDER}/artifacts/churn/"
    boto_sess = boto3.Session(region_name=AWS_REGION)
    sm_sess = Session(boto_session=boto_sess)
    IMAGE_URI = retrieve(
        framework="sklearn",
        region=AWS_REGION,
        version="1.2-1",
        py_version="py3",
        instance_type=PROCESSING_INSTANCE_TYPE,
    )

    processor = ScriptProcessor(
        image_uri=IMAGE_URI,
        command=["python3"],
        role=ROLE_ARN,
        instance_type=PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=sm_sess,
        max_runtime_in_seconds=3600,
    )

    processor.run(
        code="src/model/train_xgb.py",
        arguments=[
            "--input",
            "/opt/ml/processing/input/customer_features.parquet",
            "--out",
            "/opt/ml/processing/output",
            "--recency_threshold",
            "3",
        ],
        inputs=[
            ProcessingInput(
                input_name="input",
                s3_input=ProcessingS3Input(
                    s3_uri=input_s3,
                    local_path="/opt/ml/processing/input",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated",
                ),
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="out",
                s3_output=ProcessingS3Output(
                    s3_uri=output_s3,
                    local_path="/opt/ml/processing/output",
                    s3_upload_mode="EndOfJob",
                ),
            )
        ],
        wait=True,
        logs=True,
    )


if __name__ == "__main__":
    run_processing_job()
