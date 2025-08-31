import argparse, sagemaker
from sagemaker.huggingface import HuggingFaceModel

parser = argparse.ArgumentParser()
parser.add_argument("--role-arn", type=str)
parser.add_argument("--model-data", type=str, required=True)  # s3://.../model.tar.gz from training
parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge")
parser.add_argument("--use_custom_handlers", action="store_true")
args = parser.parse_args()

role = args.role_arn or sagemaker.get_execution_role()

version = {"transformers_version": "4.43", "pytorch_version": "2.3", "py_version": "py310"}

env = {"HF_TASK": "image-to-text"} if not args.use_custom_handlers else {}

model = HuggingFaceModel(
    role=role,
    model_data=args.model_data,
    # custom handlers
    entry_point="inference.py" if args.use_custom_handlers else None,
    source_dir="sagemaker" if args.use_custom_handlers else None,
    env=env,
    **version,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=args.instance_type,
    endpoint_name="blip-captioning-endpoint",
)
print("Endpoint:", predictor.endpoint_name)
