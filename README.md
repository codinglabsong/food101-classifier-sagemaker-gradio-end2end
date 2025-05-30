# MNIST-CNN • SageMaker Local ↔ Cloud Template

A minimal, repeatable workflow:

1. **Create & activate** your Conda env:
   ```bash
   conda env create -f environment.yml
   conda activate mnist-cnn
   ```
   *Activate your Conda env before running any make target.*

2. **Smoke-test locally** on CPU against a tiny dataset:
   ```bash
   make smoke
   ```

3. **Full GPU run** in SageMaker:
   ```bash
   make train
   ```

3. **View logs** (in another shell):
   ```bash
   make logs JOB=<your-training-job-name>
   ```

4. **Lint** your code:
   ```bash
   make lint
   ```

If you ever need to update dependencies, edit environment.yml and run:

```bash
conda env update -f environment.yml
```

Zip and upload data to S3 bucket
```bash
tar czf mnist-full.tar.gz -C data/full .
aws s3 cp mnist-full.tar.gz s3://<YOUR-BUCKET>/full/
```
