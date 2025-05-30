# Food101-classifier

Zip and upload data to S3 bucket
```bash
tar czf mnist-full.tar.gz -C data/full .
aws s3 cp mnist-full.tar.gz s3://<YOUR-BUCKET>/full/
```