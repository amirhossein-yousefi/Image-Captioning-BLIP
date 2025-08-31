import os, argparse, sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session

parser = argparse.ArgumentParser()
parser.add_argument("--role-arn", type=str, help="IAM role for SageMaker")
parser.add_argument("--region", type=str, default=os.getenv("AWS_REGION", "us-east-1"))
parser.add_argument("--bucket", type=str, required=True)
parser.add_argument("--prefix", type=str, default="blip-captioning")
parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge")
args = parser.parse_args()

sess = sagemaker.Session()
role = args.role_arn or sagemaker.get_execution_role()

# ⚠️ Check available DLC combos and adjust these if needed
# https://huggingface.co/docs/sagemaker/en/dlcs/available
HUGGINGFACE_VERSION = {
    "transformers_version": "4.43",
    "pytorch_version": "2.3",
    "py_version": "py310",
}

# S3 input channels: expect JSONL + images under these prefixes
# s3://<bucket>/<prefix>/data/train/{train.jsonl, images/...}
# s3://<bucket>/<prefix>/data/validation/{validation.jsonl, images/...}
train_s3 = f"s3://{args.bucket}/{args.prefix}/data/train"
val_s3   = f"s3://{args.bucket}/{args.prefix}/data/validation"

hyperparameters = {
    # Model/Data
    "model_name_or_path": "Salesforce/blip-image-captioning-base",
    "train_file": "train.jsonl",       # resolved relative to SM_CHANNEL_TRAIN
    "validation_file": "validation.jsonl",
    "image_root": "images",            # resolved relative to SM_CHANNEL_TRAIN
    "max_length": 64,
    # Trainer
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "fp16": True,
    "evaluation_strategy": "steps",
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "predict_with_generate": True,
}

estimator = HuggingFace(
    entry_point="train_blip.py",
    source_dir="sagemaker",
    role=role,
    instance_type=args.instance_type,
    instance_count=1,
    **HUGGINGFACE_VERSION,
    hyperparameters=hyperparameters,
    # Cost & resiliency
    use_spot_instances=True,
    max_wait=36000,  # seconds
    max_run=36000,
    enable_sagemaker_metrics=True,
    checkpoint_s3_uri=f"s3://{args.bucket}/{args.prefix}/checkpoints/",
)

inputs = {
    "train": TrainingInput(train_s3, content_type="application/json"),
    "validation": TrainingInput(val_s3, content_type="application/json"),
}

estimator.fit(inputs, job_name=f"blip-train-{sagemaker.utils.sagemaker_timestamp()}")
print("Model artifacts:", estimator.model_data)
