# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.7.11 ('markuplmft')
#     language: python
#     name: python3
# ---

import wandb
import pprint

defaults = dict(
    learning_rate=1e-5, 
    adam_epsilon=1e-8
)

# +
wandb.init(project="LanguageModel", config=defaults, resume=True)

wandb.login()
config = dict(wandb.config)
print("Configurations from WandB: ")
print(config)
# -

learning_rate = config["learning_rate"]
adam_epsilon = config["adam_epsilon"]

# +
from markuplmft.fine_tuning.run_swde.data_reader import DataReader

data_reader_config = dict(
    overwrite_cache=False,
    parallelize=False, 
    verbose=False)
dr = DataReader(**data_reader_config)

# #?  Debug
train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=2)
develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=2)

# #?  I will use 24 websites to train and 8 websites to evaluate
# train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=24)
# develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=8)

# # ? Using half of the data just to speed up for few improvements
# train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=12)
# develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=4)

# #?  Generate all features
# train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=False)
# develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=False)
# -

print(f"train_dataset_info: {len(train_dataset_info[0])}")
print(f"develop_dataset_info: {len(develop_dataset_info[0])}")

# # Train

# +
from markuplmft.fine_tuning.run_swde.markuplmodel import MarkupLModel

markup_model = MarkupLModel()

markup_model.load_pretrained_model_and_tokenizer(-1)
# markup_model.tokenizer
# markup_model.model

# +
from markuplmft.fine_tuning.run_swde.trainer import Trainer

trainer_config = dict(
    # # ? Optimizer
    weight_decay= 0.0, 
    learning_rate= learning_rate,
    adam_epsilon=adam_epsilon,

    # # ? Scheduler
    warmup_ratio=0.0,

    verbose=False
    )

per_gpu_train_batch_size = 16
eval_batch_size = 256
num_epochs = 3
max_steps=0
logging_every_epoch=1
trainer_config = trainer_config
gradient_accumulation_steps=1 # For the short test I did, increasing this doesn't change the time and reduce performance
fp16=True
no_cuda=False
fp16_opt_level="O1"
max_grad_norm=1.0
evaluate_during_training = False

save_model_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models"
overwrite_model=True

trainer = Trainer(
    model=markup_model,
    no_cuda=no_cuda,
    train_dataset_info=train_dataset_info,
    evaluate_dataset_info=develop_dataset_info,
    per_gpu_train_batch_size=per_gpu_train_batch_size,
    eval_batch_size=eval_batch_size,
    num_epochs=num_epochs,
    max_steps=max_steps,
    logging_every_epoch=logging_every_epoch,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_grad_norm=max_grad_norm,
    save_model_path=save_model_path,
    overwrite_model=overwrite_model,
    fp16=fp16,
    fp16_opt_level=fp16_opt_level,
    evaluate_during_training=evaluate_during_training,
    **trainer_config,
)
# -

dataset_nodes_predicted = trainer.train()

# # Infer

# +
# from markuplmft.fine_tuning.run_swde.markuplmodel import MarkupLModel

# markup_model = MarkupLModel()
# markup_model.load_trained_model(
#     config_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
#     tokenizer_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
#     net_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/epochs_3/checkpoint-3",
# )
# markup_model

# +
# from markuplmft.fine_tuning.run_swde.trainer import Trainer

# trainer_config = dict(
#     # ? Optimizer
#     weight_decay= 0.0, 
#     learning_rate= 1e-5,
#     adam_epsilon=1e-8,

#     # ? Scheduler
#     warmup_ratio=0.0,

#     verbose=False
#     )

# per_gpu_train_batch_size = 16
# eval_batch_size = 64
# num_epochs = 1
# max_steps=0
# logging_every_epoch=1
# trainer_config = trainer_config
# gradient_accumulation_steps=1 # For the short test I did, increasing this doesn't change the time and reduce performance
# fp16=True
# no_cuda=False
# fp16_opt_level="O1"
# max_grad_norm=1.0
# evaluate_during_training = False

# save_model_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models"
# overwrite_model=True

# trainer = Trainer(
#     model=markup_model,
#     no_cuda=no_cuda,
#     train_dataset_info=train_dataset_info,
#     evaluate_dataset_info=train_dataset_info,
#     per_gpu_train_batch_size=per_gpu_train_batch_size,
#     eval_batch_size=eval_batch_size,
#     num_epochs=num_epochs,
#     max_steps=max_steps,
#     logging_every_epoch=logging_every_epoch,
#     gradient_accumulation_steps=gradient_accumulation_steps,
#     max_grad_norm=max_grad_norm,
#     save_model_path=save_model_path,
#     overwrite_model=overwrite_model,
#     fp16=fp16,
#     fp16_opt_level=fp16_opt_level,
#     evaluate_during_training=evaluate_during_training,
#     **trainer_config,
# )

# +
# dataset_nodes_predicted = trainer.evaluate(trainer.evaluate_dataloader, trainer.evaluate_dataset, trainer.evaluate_info)
# -


