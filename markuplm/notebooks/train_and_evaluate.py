# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.7.11 ('markuplmft')
#     language: python
#     name: python3
# ---

# %%
from markuplmft.fine_tuning.run_swde.data_reader import DataReader

config = dict(
    overwrite_cache=True,
    save_features=True,
    parallelize=False, 
    verbose=False)
dr = DataReader(**config)

# train_dataset, train_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=3, to_evaluate=False)
# develop_dataset, develop_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=3, to_evaluate=True)


# #! Trying to use only develop_dataset_info because it has info and loss 
# train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=3, to_evaluate=True)
develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=3, to_evaluate=True)

# %%
from markuplmft.fine_tuning.run_swde.markuplmodel import MarkupLModel

markup_model = MarkupLModel()

# %%
markup_model.load_pretrained_tokenizer()
# markup_model.tokenizer

# %%
markup_model.load_pretrained_model(-1)
# markup_model.model

# %%
from markuplmft.fine_tuning.run_swde.trainer import Trainer

trainer_config = dict(
    # # ? Optimizer
    weight_decay= 0.0, 
    learning_rate= 1e-5,
    adam_epsilon=1e-8,

    # # ? Scheduler
    warmup_ratio=0.0,

    verbose=False
    )

model = markup_model.net
per_gpu_train_batch_size = 16
eval_batch_size = 16
num_epochs = 1
max_steps=20
logging_every_epoch=1
trainer_config = trainer_config
gradient_accumulation_steps=1
output_dir="/data/GIT/unilm/markuplm/markuplmft/models/markuplm/"
fp16=True
no_cuda=False
fp16_opt_level="O1"
overwrite_output_dir=True
max_grad_norm=1.0
evaluate_during_training = False

trainer = Trainer(
    model=model,
    no_cuda=no_cuda,
    train_dataset_info=develop_dataset_info,
    develop_dataset_info=develop_dataset_info,
    per_gpu_train_batch_size=per_gpu_train_batch_size,
    eval_batch_size=eval_batch_size,
    num_epochs=num_epochs,
    max_steps=max_steps,
    logging_every_epoch=logging_every_epoch,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_grad_norm=max_grad_norm,
    output_dir=output_dir,
    fp16=fp16,
    fp16_opt_level=fp16_opt_level,
    overwrite_output_dir=overwrite_output_dir,
    evaluate_during_training=evaluate_during_training,
    **trainer_config,
)

# %%
dataset_nodes_predicted = trainer.train()

# %%
# dataset_nodes_predicted

# %%
# train_dataloader = trainer._get_dataloader(trainer.train_dataset, True)
# eval_dataloader = trainer._get_dataloader(trainer.develop_dataset, False)

# %%
# for x in train_dataloader:
#     print(len(x))
# # for x in eval_dataloader:
# #     print(len(x))

# %%
# from markuplmft.fine_tuning.run_swde.lm_model import LModel
# lm = LModel()
# lm.parallelize = False
# lm.load_data(dataset='train', limit_data=10)

# %%
# lm.load_data(dataset='develop', limit_data=10)
# lm.prepare_model_to_train()
# lm.fit()
