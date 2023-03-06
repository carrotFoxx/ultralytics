import wandb
from wandb.sdk.data_types import *
api = wandb.Api()
project_name = "navi"
dataset_name = "paper_cup"
api_key = "5271adad4bff1f2f00d405fe8fae1dfa993f994a"

# Авторизация в API Weights & Biases
wandb.login(key=api_key)

# Загрузка датасета
run = wandb.init()
artifact = wandb.Artifact(dataset_name, type="dataset")
artifact.add_dir("datasets/paper_cup")
run.log_artifact(artifact)

# Публикация датасета
run.finish()
artifact.wait()
