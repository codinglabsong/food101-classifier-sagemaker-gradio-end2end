import os, yaml
from dotenv import load_dotenv 
from sagemaker.pytorch import PyTorch
load_dotenv()

cfg = yaml.safe_load(open("config/prod.yaml"))

est = PyTorch(
    entry_point        = "train.py",
    source_dir         = "src",
    role               = os.getenv("ROLE_ARN"),
    instance_count     = cfg["estimator"]["instance_count"],
    instance_type      = cfg["estimator"]["instance_type"],
    framework_version  = cfg["estimator"]["framework_version"],
    py_version         = cfg["estimator"]["py_version"],
    base_job_name      = cfg["estimator"]["base_job_name"],
    use_spot_instances = cfg["estimator"]["use_spot_instances"],
    max_run            = cfg["estimator"]["max_run"],
    max_wait           = cfg["estimator"]["max_wait"], 
    environment        = {"WANDB_API_KEY": os.getenv("WANDB_API_KEY")}, 
    dependencies       = ['requirements.txt'],
    hyperparameters    = cfg["estimator"]["hyperparameters"],
)

bucket  = os.getenv("SM_BUCKET")
train_uri = f"s3://{bucket}/full/food101-train.tar.gz"
test_uri = f"s3://{bucket}/full/food101-test.tar.gz"

est.fit({"train": train_uri, "test": test_uri})