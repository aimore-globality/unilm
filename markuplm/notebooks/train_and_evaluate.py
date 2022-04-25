# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import wandb
import os
from pprint import pprint
import torch

from markuplmft.fine_tuning.run_swde.utils import get_device_and_gpu_count

from markuplmft.fine_tuning.run_swde.markuplmodel import MarkupLModel


# %%
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank=-1

print(f"local_rank: {local_rank}")

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# %%
no_cuda = False
device, n_gpu = get_device_and_gpu_count(no_cuda, local_rank)


# %%
trainer_config = dict(
    # # ? Optimizer
    weight_decay= 0.01, #? Default: 0.0
    learning_rate=1e-05,  #? Default: 1e-05
    adam_epsilon=1e-8, #? Default: 1e-8
    # # ? Loss
    label_smoothing=0.01, #? Default: 0.0 
    loss_function = "CrossEntropyLoss", #? Default: CrossEntropyLoss / FocalLoss
    # # ? Scheduler
    warmup_ratio=0.0, #? Default: 0
    # # ? Trainer
    num_epochs = 80, 
    logging_every_epoch = 1,
    gradient_accumulation_steps = 1, #? For the short test I did, increasing this doesn't change the time and reduce performance
    max_steps = 0, 
    fp16 = True, 
    fp16_opt_level = "O1",
    max_grad_norm = 1.0,
    verbose = False,
    save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models",
    per_gpu_train_batch_size = 34, #? 34 Max with the big machine 
    # per_gpu_train_batch_size = 16, #? Max with the big machine 
    eval_batch_size = 1024, #? 1024 Max with the big machine 
    # eval_batch_size = 128, #?  Max with the big machine 
    overwrite_model = True,
    evaluate_during_training = True,
    no_cuda = no_cuda,
    freeze_body = False,
    dataset_to_use='all',
    # # ? Data Reader
    overwrite_cache=True, 
    parallelize=False, 
    train_dedup=True, #? Default: False
    develop_dedup=True, #? Default: False
)
if trainer_config['dataset_to_use'] == 'all': trainer_config["parallelize"] = True
if trainer_config['dataset_to_use'] == 'debug': trainer_config["num_epochs"] = 1


# %%
print(f"local_rank: {local_rank}")

if local_rank in [-1, 0]:
    print("Initializing WandB...")
    run = wandb.init(project="LanguageModel", config=trainer_config, resume=False)
    trainer_config = dict(run.config)
    print("Training configurations from WandB: ")
    pprint(trainer_config)
else:
    run = None
    
loss_function = trainer_config.pop("loss_function")
label_smoothing = trainer_config.pop("label_smoothing")
# %%
from markuplmft.fine_tuning.run_swde.data_reader import DataReader

dr = DataReader(
    local_rank=local_rank, 
    overwrite_cache=trainer_config.pop("overwrite_cache"), 
    parallelize=trainer_config.pop("parallelize"),
    verbose=trainer_config.get("verbose"), 
    n_gpu=n_gpu)

dataset_to_use = trainer_config.pop("dataset_to_use", "debug")

if trainer_config.pop("train_dedup"):
    train_dedup = "_dedup"
else:
    train_dedup = ""
if trainer_config.pop("develop_dedup"):
    develop_dedup = "_dedup"
else:
    develop_dedup = ""

# #?  Debug
if dataset_to_use == "debug":
    train_dataset_info = dr.load_dataset(data_dir=f"/data/GIT/swde/my_data/train/my_CF_processed{train_dedup}/", limit_data=2)
    develop_dataset_info = dr.load_dataset(data_dir=f"/data/GIT/swde/my_data/develop/my_CF_processed{develop_dedup}/", limit_data=2)

# #?  I will use 24 websites to train and 8 websites to evaluate
elif dataset_to_use == "mini":
    train_dataset_info = dr.load_dataset(data_dir=f"/data/GIT/swde/my_data/train/my_CF_processed{train_dedup}/", limit_data=24)
    develop_dataset_info = dr.load_dataset(data_dir=f"/data/GIT/swde/my_data/develop/my_CF_processed{develop_dedup}/", limit_data=8)

# #?  Generate all features
elif dataset_to_use == "all":
    train_dataset_info = dr.load_dataset(data_dir=f"/data/GIT/swde/my_data/train/my_CF_processed{train_dedup}/", limit_data=False)
    develop_dataset_info = dr.load_dataset(data_dir=f"/data/GIT/swde/my_data/develop/my_CF_processed{develop_dedup}/", limit_data=False)
else:
    pass

# %%
print(f"train_dataset_info: {len(train_dataset_info[0])}")
print(f"develop_dataset_info: {len(develop_dataset_info[0])}")

# %% [markdown]
# # Train

# %%
print(f"\n local_rank: {local_rank} - Loading pretrained model and tokenizer...")
markup_model = MarkupLModel(local_rank=local_rank, loss_function=loss_function, label_smoothing=label_smoothing, device=device, n_gpu=n_gpu)
markup_model.load_pretrained_model_and_tokenizer()

if trainer_config.pop("freeze_body", False):
    markup_model.freeze_body()

# %%
from markuplmft.fine_tuning.run_swde.trainer import Trainer

print(f"\n{local_rank} - Preparing Trainer...")
# #? Leave this barrier here because it unlocks
# #? the other GPUs that were waiting at: 
# #? load_or_cache_websites in DataReader
if local_rank == 0: 
    torch.distributed.barrier()

trainer = Trainer(
    model = markup_model,
    train_dataset_info = train_dataset_info,
    evaluate_dataset_info = develop_dataset_info,
    local_rank=local_rank,
    device=device, 
    n_gpu=n_gpu,
    run=run,
    **trainer_config,
)

# %%
print(f"\nTraining...")
dataset_nodes_predicted = trainer.train()

# %%
dataset_nodes_predicted

# %% [markdown]
# # Infer

# %%
if local_rank not in [-1, 0]:
    torch.distributed.barrier()

if local_rank in [-1, 0]:
    load_model_path = markup_model.save_path
    # load_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models/epochs_3/checkpoint-3"
    
    del markup_model
    torch.cuda.empty_cache()

    from markuplmft.fine_tuning.run_swde.markuplmodel import MarkupLModel

    print(f"load_model_path: {load_model_path}")

    trained_markup_model = MarkupLModel(local_rank=local_rank, loss_function=loss_function, label_smoothing=label_smoothing, device=device, n_gpu=n_gpu)
    trained_markup_model.load_trained_model(
        config_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
        tokenizer_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
        net_path=load_model_path,
    )
    # print(trained_markup_model)

    trainer = Trainer(
        model = trained_markup_model,
        train_dataset_info = train_dataset_info,
        evaluate_dataset_info = develop_dataset_info,
        local_rank=local_rank,
        device=device, 
        n_gpu=n_gpu,
        run=run,
        just_evaluation=True,
        **trainer_config,
    )

    train_set_nodes_predicted = trainer.evaluate(dataset_name="train")
    print(f"Train dataset predicted size: {len(train_set_nodes_predicted)}")

    develop_set_nodes_predicted = trainer.evaluate(dataset_name="develop")
    print(f"Develop dataset predicted size: {len(develop_set_nodes_predicted)}")

    save_path = f"develop_set_nodes_classified_epoch_{trainer_config['num_epochs']}.pkl"
    print(save_path)
    develop_set_nodes_predicted.to_pickle(save_path)

    run.save()
    run.finish()

if local_rank not in [-1, 0]:
    torch.distributed.barrier()

# %%

# %%
