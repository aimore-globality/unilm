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
import glob
import pandas as pd
import transformers
from markuplmft.fine_tuning.run_swde.featurizer import Featurizer
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import wandb
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.utils import set_seed
from transformers import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AdamW, get_scheduler
from transformers import set_seed
import torch
from transformers import get_linear_schedule_with_warmup
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset



# %%
trainer_config = dict(
    dataset_to_use='all',
    train_dedup=True, #? Default: False
    develop_dedup=True, #? Default: False
)


# %%
if trainer_config.pop("train_dedup"):
    train_dedup = "_dedup"
else:
    train_dedup = ""

if trainer_config.pop("develop_dedup"):
    develop_dedup = "_dedup"
else:
    develop_dedup = ""

train_domains_path = glob.glob(f"/data/GIT/delete/train/processed{train_dedup}/*.pkl")
develop_domains_path = glob.glob(f"/data/GIT/delete/develop/processed{develop_dedup}/*.pkl")

print(f"train_domains_path: {len(train_domains_path)} - {train_domains_path[0]}")
print(f"develop_domains_path: {len(develop_domains_path)} - {develop_domains_path[0]}")

dataset_to_use = trainer_config.pop("dataset_to_use")

# #?  I will use 24 websites to train and 8 websites to evaluate
if dataset_to_use == "mini":
    train_domains_path = train_domains_path[:24]
    develop_domains_path = develop_domains_path[:8]

# #?  Generate all features
elif dataset_to_use == "all":
    train_domains_path = train_domains_path
    develop_domains_path = develop_domains_path

# #?  Debug
else:
    train_domains_path = train_domains_path[:4]
    develop_domains_path = develop_domains_path[:4]

df_train = pd.DataFrame()
for domain_path in train_domains_path:
    df_train = df_train.append(pd.read_pickle(domain_path)) 

df_develop = pd.DataFrame()
for domain_path in develop_domains_path:
    df_develop = df_develop.append(pd.read_pickle(domain_path)) 

# %%
print(f"train_dataset: {len(df_train)}")
print(f"develop_dataset: {len(df_develop)}")

# %% [markdown]
# # Train

# %%
# # ? Trainer
num_epochs = 4
train_batch_size = 30  #? 34 Max with the big machine 
evaluate_batch_size = 8*train_batch_size #? 1024 Max with the big machine 
# # ? Setting Data
train_dataset = df_train
evaluate_dataset = df_develop

train_dataset["html"] = train_dataset["html"].astype("category")
evaluate_dataset["html"] = evaluate_dataset["html"].astype("category")

num_train_epochs = num_epochs
num_training_steps = num_epochs

# %%
from sklearn.metrics import recall_score, precision_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def create_data_loader(features, batch_size, is_train=True):
    return DataLoader(
        dataset=features,
        shuffle=is_train,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4
    )

def training_function(evaluate_dataset):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        
    featurizer = Featurizer()
    
    train_features = featurizer.feature_to_dataset(train_dataset["swde_features"].explode().values)
    evaluate_features = featurizer.feature_to_dataset(evaluate_dataset["swde_features"].explode().values)

    train_dataloader = create_data_loader(train_features, train_batch_size, True)
    evaluate_dataloader = create_data_loader(evaluate_features, evaluate_batch_size, False)

    set_seed(0)

    train_batches = len(train_dataloader)
    evaluate_batches = len(evaluate_dataloader)

    accelerator.print(f"batch_size = {train_batch_size} | train_batches: {train_batches} | training_samples: {len(train_features)}")
    accelerator.print(f"batch_size = {evaluate_batch_size} | evaluate_batches: {evaluate_batches} | evaluate_samples: {len(evaluate_features)}")

    accelerator.print(f"Num Epochs = {num_train_epochs}")
    accelerator.print(f"Num training data points = {len(train_dataset)}")

    device = accelerator.device
    accelerator.print(f"device: {device}")

    model = transformers.RobertaForTokenClassification.from_pretrained('roberta-base')
    model.to(device)

    weight_decay =  0.01 #? Default: 0.0
    learning_rate = 1e-05  #? Default: 1e-05
    adam_epsilon = 1e-8 #? Default: 1e-8
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                param
                for param_name, param in model.named_parameters()
                if not any(nd in param_name for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param
                for param_name, param in model.named_parameters()
                if any(nd in param_name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    (
        model, optimizer, train_dataloader, evaluate_dataloader 
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, evaluate_dataloader, 
    )
    
    scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * num_epochs
        )

    # #! Training step
    accelerator.print("Train...")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    for epoch in progress_bar:
        accelerator.print(f"Epoch: {epoch}")
        all_losses = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],  
                "labels": batch[3],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            all_losses.append(loss.item())
        
        accelerator.print(f"Loss: {np.mean(all_losses)}")
        progress_bar.update(1)

    # #! Evaluation step
    accelerator.print("Evaluate...")
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    for step, batch in enumerate(evaluate_dataloader):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],  
                "labels": batch[3],
            }
            outputs = model(**inputs)
        logits = outputs.logits
        all_logits.append(accelerator.gather(logits).detach().cpu())

    #     predictions = outputs.logits.argmax(dim=-1)
    #     all_predictions.append(accelerator.gather(predictions))
    #     all_labels.append(accelerator.gather(inputs["labels"]))

    # all_predictions = torch.cat(all_predictions)[:len(evaluate_features)]
    # all_labels = torch.cat(all_labels)[:len(evaluate_features)]

    # all_labels, all_predictions = np.array(all_labels.cpu()), np.array(all_predictions.cpu())
    # all_labels, all_predictions = np.clip(all_labels, a_min = 0, a_max=1).reshape(-1,1), np.clip(all_predictions, a_min = 0, a_max=1).reshape(-1,1)
    # p = precision_score(all_labels, all_predictions)
    # r = recall_score(all_labels, all_predictions)
    # accelerator.print(f"Final: P: {p:.2f} R:{r:.2f}")
    # eval_metric = metric.compute(predictions=all_predictions, references=all_labels)

    all_probs = torch.softmax(torch.cat(all_logits, dim=0), dim=2)
    node_probs = []

    for feature_index, feature_ids in enumerate(evaluate_features.relative_first_tokens_node_index):
        node_probs.extend(all_probs[feature_index, [evaluate_features.relative_first_tokens_node_index[feature_index]], 0][0])

    node_probs = np.array(node_probs)
    accelerator.print(len(node_probs))
    accelerator.print("dataset: ", len(evaluate_dataset))
    evaluate_dataset = evaluate_dataset.explode('nodes', ignore_index=True).reset_index()
    evaluate_dataset = evaluate_dataset.join(pd.DataFrame(evaluate_dataset.pop('nodes').tolist(), columns=["xpath","node_text","node_gt_tag","node_gt_text"]))
    accelerator.print(f"Memory: {sum(evaluate_dataset.memory_usage(deep=True))/10**6:.2f} Mb")
    evaluate_dataset.drop(['html', "swde_features"], axis=1, inplace=True)
    accelerator.print(f"Memory: {sum(evaluate_dataset.memory_usage(deep=True))/10**6:.2f} Mb")
    evaluate_dataset['node_prob'] = node_probs
    evaluate_dataset['node_pred'] = node_probs > 0.5

    # TODO: move this out        
    evaluate_dataset["node_gt"] = evaluate_dataset["node_gt_tag"] == 'PAST_CLIENT'
    evaluate_dataset["node_pred_tag"] = evaluate_dataset["node_pred"].apply(lambda x: "PAST_CLIENT" if x else "none")

    def get_classification_metrics(dataset_predicted):
        accelerator.print("Compute Metrics:")
        metrics_per_dataset, cm_per_dataset = compute_metrics_per_dataset(dataset_predicted)

        accelerator.print(
            f"Node Classification Metrics per Dataset:\n {metrics_per_dataset} | cm_per_dataset: {cm_per_dataset}"
        )
        return metrics_per_dataset, cm_per_dataset

    metrics_per_dataset, cm_per_dataset = get_classification_metrics(evaluate_dataset)
    accelerator.print(f"metrics_per_dataset: {metrics_per_dataset}")
    accelerator.print(f"cm_per_dataset: {cm_per_dataset}")
    accelerator.print("...Done")


# %%

from accelerate import notebook_launcher
notebook_launcher(training_function, args=[evaluate_dataset],  num_processes=4, use_fp16=True, mixed_precision="bf16")

# %%
pd.set_option("max_columns", 200)
dataset_nodes_predicted.head(2)

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
# if local_rank not in [-1, 0]:
#     torch.distributed.barrier()

# if local_rank in [-1, 0]:
#     load_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models/epochs_3/checkpoint-3"
    
#     del model
#     torch.cuda.empty_cache()

#     print(f"load_model_path: {load_model_path}")

#     # trained_model = MarkupLModel(local_rank=local_rank, loss_function=loss_function, label_smoothing=label_smoothing, device=device, n_gpu=n_gpu)
#     trained_model  = transformers.RobertaForTokenClassification.from_pretrained(trainer_config.save_model_path)
#     # trained_model.load_trained_model(
#     #     config_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
#     #     tokenizer_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
#     #     net_path=load_model_path,
#     # )
#     print(trained_model)

#     trainer = Trainer(
#         model = trained_model,
#         train_dataset = df_train,
#         evaluate_dataset = df_develop,
#         featurizer=featurizer,
#         local_rank=local_rank,
#         device=device, 
#         n_gpu=n_gpu,
#         just_evaluation=True,
#         run=run,
#         **trainer_config,
#     )

#     train_nodes_predicted = trainer.evaluate(dataset_name="train")
#     print(f"Train dataset predicted size: {len(train_nodes_predicted)}")

#     save_path = f"results_classified/train_set_nodes_classified_epoch_{trainer_config['num_epochs']}{train_dedup}.pkl"
#     print(f"Data infered saved at: {save_path}")
#     train_nodes_predicted.to_pickle(save_path)
    

#     develop_nodes_predicted = trainer.evaluate(dataset_name="develop")
#     print(f"Develop dataset predicted size: {len(develop_nodes_predicted)}")
    
#     save_path = f"results_classified/develop_set_nodes_classified_epoch_{trainer_config['num_epochs']}{develop_dedup}.pkl"
#     print(f"Data infered saved at: {save_path}")
#     develop_nodes_predicted.to_pickle(save_path)

#     run.save()
#     run.finish()

# if local_rank not in [-1, 0]:
#     torch.distributed.barrier()

# %%
# save_path

# %%
run.save()
run.finish()
