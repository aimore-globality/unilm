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
import glob
import pandas as pd

from markuplmft.fine_tuning.run_swde.utils import get_device_and_gpu_count
import transformers
from transformers import RobertaTokenizer
from markuplmft.fine_tuning.run_swde.featurizer import Featurizer



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
    num_epochs = 4, 
    gradient_accumulation_steps = 1, #? For the short test I did, increasing this doesn't change the time and reduce performance
    max_steps = 0, 
    # per_gpu_train_batch_size = int(34), #? 34 Max with the big machine 
    per_gpu_train_batch_size = int(16), #? 34 Max with the big machine 
    # eval_batch_size = int(1024), #? 1024 Max with the big machine 
    eval_batch_size = int(128), #? 1024 Max with the big machine 
    fp16 = True, 
    fp16_opt_level = "O1",
    max_grad_norm = 1.0,
    # load_model=False,
    # load_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models/epochs_2/checkpoint-2",
    # freeze_body = False,
    save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models",
    overwrite_model = True,
    evaluate_during_training = True,
    no_cuda = no_cuda,
    verbose = False,
    logging_every_epoch = 1,
    # # ? Data Reader
    dataset_to_use='debug',
    # parallelize=False, 
    train_dedup=False, #? Default: False
    develop_dedup=False, #? Default: False
)
if trainer_config['dataset_to_use'] == 'all': trainer_config["parallelize"] = True
if trainer_config['dataset_to_use'] == 'debug': trainer_config["num_epochs"] = 1


# %%
# # trainer_config = dict(
# #     # ? Optimizer
# #     weight_decay= 0.01, #? Default: 0.0
# #     learning_rate=1e-05,  #? Default: 1e-05
# #     adam_epsilon=1e-8, #? Default: 1e-8
# #     # ? Loss
# #     label_smoothing=0.01, #? Default: 0.0 
# #     loss_function = "CrossEntropyLoss", #? Default: CrossEntropyLoss / FocalLoss
# #     # ? Scheduler
# #     warmup_ratio=0.0, #? Default: 0
# #     # ? Trainer
# #     num_epochs = 3, 
# #     logging_every_epoch = 1,
# #     gradient_accumulation_steps = 1, #? For the short test I did, increasing this doesn't change the time and reduce performance
# #     max_steps = 0, 
# #     fp16 = True, 
# #     fp16_opt_level = "O1",
# #     max_grad_norm = 1.0,
# #     verbose = False,
# #     load_model=False,
# #     load_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models/epochs_2/checkpoint-2",
# #     save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models",
# #     per_gpu_train_batch_size = 34, #? 34 Max with the big machine 
# #     # per_gpu_train_batch_size = 16, #? Max with the big machine 
# #     eval_batch_size = 1024, #? 1024 Max with the big machine 
# #     # eval_batch_size = 128, #?  Max with the big machine 
# #     overwrite_model = True,
# #     evaluate_during_training = True,
# #     no_cuda = no_cuda,
# #     freeze_body = False,
# #     dataset_to_use='all',
# #     # ? Data Reader
# #     overwrite_cache=False, 
# #     parallelize=False, 
# #     train_dedup=True, #? Default: False
# #     develop_dedup=True, #? Default: False
# # )
# # if trainer_config['dataset_to_use'] == 'all': trainer_config["parallelize"] = True
# # if trainer_config['dataset_to_use'] == 'debug': trainer_config["num_epochs"] = 1

# # #! Train only head after being trained full body:
# trainer_config = dict(
#     # ? Optimizer
#     weight_decay= 0.01, #? Default: 0.0
#     learning_rate=1e-05,  #? Default: 1e-05
#     adam_epsilon=1e-8, #? Default: 1e-8
#     # ? Loss
#     label_smoothing=0.01, #? Default: 0.0 
#     loss_function = "FocalLoss", #? Default: CrossEntropyLoss / FocalLoss
#     # ? Scheduler
#     warmup_ratio=0.0, #? Default: 0
#     # ? Trainer
#     num_epochs = 10, 
#     logging_every_epoch = 1,
#     gradient_accumulation_steps = 1, #? For the short test I did, increasing this doesn't change the time and reduce performance
#     max_steps = 0, 
#     fp16 = True, 
#     fp16_opt_level = "O1",
#     max_grad_norm = 1.0,
#     verbose = False,
#     load_model=True,
#     load_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models/epochs_2/checkpoint-2",
#     save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models",
#     per_gpu_train_batch_size = 34, #? 34 Max with the big machine 
#     # per_gpu_train_batch_size = 16, #? Max with the big machine 
#     eval_batch_size = 1024, #? 1024 Max with the big machine 
#     # eval_batch_size = 128, #?  Max with the big machine 
#     overwrite_model = True,
#     evaluate_during_training = True,
#     no_cuda = no_cuda,
#     freeze_body = True,
#     dataset_to_use='all',
#     # ? Data Reader
#     overwrite_cache=False, 
#     parallelize=False, 
#     train_dedup=True, #? Default: False
#     develop_dedup=True, #? Default: False
#     seed=,
# )
# if trainer_config['dataset_to_use'] == 'all': trainer_config["parallelize"] = True
# if trainer_config['dataset_to_use'] == 'debug': trainer_config["num_epochs"] = 1


# %%
print(f"local_rank: {local_rank}")

if local_rank in [-1, 0]:
    print("Initializing WandB...")
    run = wandb.init(project="LanguageModel", config=trainer_config, resume=False, tags=["train_evaluate"])
    trainer_config = dict(run.config)
    print("Training configurations from WandB: ")
    pprint(trainer_config)
else:
    run = None

loss_function = trainer_config.pop("loss_function")
label_smoothing = trainer_config.pop("label_smoothing")
# %%
if trainer_config.pop("train_dedup"):
    train_dedup = "_dedup"
else:
    train_dedup = ""

if trainer_config.pop("develop_dedup"):
    develop_dedup = "_dedup"
else:
    develop_dedup = ""

train_domains_path = glob.glob(f"/data/GIT/delete/train{train_dedup}/*.pkl")
develop_domains_path = glob.glob(f"/data/GIT/delete/develop{develop_dedup}/*.pkl")

dataset_to_use = trainer_config.pop("dataset_to_use", "debug")
# #?  Debug
if dataset_to_use == "debug":
    train_domains_path = train_domains_path[:4]
    develop_domains_path = develop_domains_path[:4]

# #?  I will use 24 websites to train and 8 websites to evaluate
elif dataset_to_use == "mini":
    train_domains_path = train_domains_path[:24]
    develop_domains_path = develop_domains_path[:8]

# #?  Generate all features
elif dataset_to_use == "all":
    train_domains_path = train_domains_path
    develop_domains_path = develop_domains_path

else:
    pass

df_train = pd.DataFrame()
for domain_path in train_domains_path:
    df_train = df_train.append(pd.read_pickle(domain_path)) 

df_develop = pd.DataFrame()
for domain_path in train_domains_path:
    df_develop = df_develop.append(pd.read_pickle(domain_path)) 

# %%
print(f"train_dataset_info: {len(df_train)}")
print(f"develop_dataset_info: {len(df_develop)}")

# %% [markdown]
# # Train

# %%
# print(f"\n local_rank: {local_rank} - Loading pretrained model and tokenizer...")
# load_model_path = trainer_config.pop("load_model_path") 
# if trainer_config.pop("load_model", False):
#     print("\n --- MODEL LOADED! --- ")
#     markup_model = MarkupLModel(local_rank=local_rank, loss_function=loss_function, label_smoothing=label_smoothing, device=device, n_gpu=n_gpu)
#     markup_model.load_trained_model(
#         config_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
#         tokenizer_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
#         net_path=load_model_path,
#     )
#     print(f"load_model_path: {load_model_path}")
#     print("\n --- MODEL LOADED! --- ")
# else:
#     markup_model = MarkupLModel(local_rank=local_rank, loss_function=loss_function, label_smoothing=label_smoothing, device=device, n_gpu=n_gpu)
#     markup_model.load_pretrained_model_and_tokenizer()

# if trainer_config.pop("freeze_body", False):
#     markup_model.freeze_body()

# %%
model  = transformers.RobertaForTokenClassification.from_pretrained('roberta-base')

# %%
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

DOC_STRIDE = 128
MAX_SEQ_LENGTH = 384
featurizer = Featurizer(tokenizer=tokenizer, doc_stride=DOC_STRIDE, max_length=MAX_SEQ_LENGTH)

# %%
from markuplmft.fine_tuning.run_swde.trainer import Trainer

print(f"\n{local_rank} - Preparing Trainer...")
# #? Leave this barrier here because it unlocks
# #? the other GPUs that were waiting at: 
# #? load_or_cache_websites in DataReader
if local_rank == 0: 
    torch.distributed.barrier()

trainer = Trainer(
    model = model,
    train_dataset = df_train,
    evaluate_dataset = df_develop,
    featurizer=featurizer,
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
dataset_nodes_predicted.head(1)

# %%
from IPython.display import display
import pandas as pd
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

metrics_per_domain = pd.DataFrame(dataset_nodes_predicted.groupby("domain").apply(lambda x: compute_metrics_per_dataset(x)[0]).to_dict()).T
cm_per_domain = pd.DataFrame(dataset_nodes_predicted.groupby("domain").apply(lambda x: compute_metrics_per_dataset(x)[1]).to_dict()).T
metrics_per_domain = metrics_per_domain.join(cm_per_domain)
metrics_per_domain = metrics_per_domain.sort_values("f1", ascending=False)

full_perf_style = {
            "precision": lambda x: "{:.1f}%".format(x * 100) if x > 0  else '',
            "recall": lambda x: "{:.1f}%".format(x * 100) if x > 0  else '',
            "f1": lambda x: "{:.1f}%".format(x * 100) if x > 0 else '',
            "TP": "{:.0f}",
            "TN": "{:.0f}",
            "FP": "{:.0f}",
            "FN": "{:.0f}",
        }

metrics_per_domain.style.format(full_perf_style).to_html('metrics_per_domain.html')
run.log({"metrics_per_domain": wandb.Html(open('metrics_per_domain.html'))})

print("Metrics per domain:")
with pd.option_context("max_rows", 500, "min_rows", 500):
    display(metrics_per_domain.style.format(full_perf_style))

# %%
# r = metrics_per_domain[~metrics_per_domain.index.isin(['greatplacetowork.com'])].sum()
r = metrics_per_domain.sum()
print(f"Precision : {r['TP']/(r['TP']+r['FP']):.3f}")
print(f"Recall    : {r['TP']/(r['TP']+r['FN']):.3f}") #? 52 -> 71


# %%
def get_worst_3(metric='precision'):
    print(f"Worst 3: {metric.capitalize()}")
    worst_3 = metrics_per_domain.sort_values(metric).dropna().iloc[:3]
    display(worst_3)
    worst_3 = list(worst_3.index)
    print(worst_3)
    return worst_3

get_worst_3('precision')
get_worst_3('recall')

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

    save_path = f"results_classified/train_set_nodes_classified_epoch_{trainer_config['num_epochs']}{train_dedup}.pkl"
    print(f"Data infered saved at: {save_path}")
    train_set_nodes_predicted.to_pickle(save_path)
    

    develop_set_nodes_predicted = trainer.evaluate(dataset_name="develop")
    print(f"Develop dataset predicted size: {len(develop_set_nodes_predicted)}")
    
    save_path = f"results_classified/develop_set_nodes_classified_epoch_{trainer_config['num_epochs']}{develop_dedup}.pkl"
    print(f"Data infered saved at: {save_path}")
    develop_set_nodes_predicted.to_pickle(save_path)

    run.save()
    run.finish()

if local_rank not in [-1, 0]:
    torch.distributed.barrier()

# %%
save_path

# %%
