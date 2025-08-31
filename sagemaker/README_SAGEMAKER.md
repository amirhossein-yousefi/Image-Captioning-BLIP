# BLIP Image Captioning on Amazon SageMaker

## One-time setup
1) Create an S3 bucket (e.g., `s3://<your-bucket>/blip-captioning/`).
2) Prepare data:
   - `data/train/train.jsonl` and `data/train/images/*`
   - `data/validation/validation.jsonl` and `data/validation/images/*`
   Each JSONL row: `{"image": "images/0001.jpg", "text": "a cat on a sofa"}`.
3) Upload to S3 under the `prefix` used below.

## Train
```bash
python sagemaker/launch_training.py \
  --bucket <your-bucket> \
  --prefix blip-captioning \
  --role-arn arn:aws:iam::<acct>:role/<SageMakerExecutionRole>
```
## Deploy
```bash
# Use printed ModelArtifacts S3 URI from training
python sagemaker/deploy_endpoint.py \
  --model-data s3://<bucket>/<prefix>/output/<job>/output/model.tar.gz
```
