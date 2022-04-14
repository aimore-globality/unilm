# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
from pprint import pprint

# +
# Argparse here get
# local_rank and pass to the trainer 
# -

trainer_defaults = dict(
    # # ? Optimizer
    weight_decay= 0.0, #? Default: 0.0
    learning_rate=2e-6,  #? Default: 2e-6
    adam_epsilon=1e-8, #? Default: 1e-8
    # # ? Scheduler
    warmup_ratio=0.0, #? Default: 1e-8
    # # ? Others
    num_epochs = 10, 
    logging_every_epoch = 1,
    gradient_accumulation_steps = 1, #? For the short test I did, increasing this doesn't change the time and reduce performance
    max_steps = 0,
    fp16 = True,
    fp16_opt_level = "O1",
    max_grad_norm = 1.0,
    verbose = False,
    save_model_path = "/data/GIT/unilm/markuplm/markuplmft/models/my_models",
    per_gpu_train_batch_size = 34 ,#? 34 Max with the big machine 
    eval_batch_size = 1024, #? 1024 Max with the big machine 
    overwrite_model = True,
    evaluate_during_training = True,
    no_cuda = False,
    freeze_body = False,
    dataset_to_use='all',
)

wandb.init(project="LanguageModel", config=trainer_defaults, resume=True)
wandb.login()
trainer_config = dict(wandb.config)
print("Training configurations from WandB: ")
pprint(trainer_config)

datareader_config = dict(
    overwrite_cache=False,
    parallelize=False, 
    verbose=False)

# +
from markuplmft.fine_tuning.run_swde.data_reader import DataReader

dr = DataReader(**datareader_config)
dataset_to_use = trainer_config.pop("dataset_to_use", "debug")

# #?  Debug
if dataset_to_use == "debug":
    train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=2)
    develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=2)

# #?  I will use 24 websites to train and 8 websites to evaluate
elif dataset_to_use == "mini":
    train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=24)
    develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=8)

# #?  Generate all features
elif dataset_to_use == "all":
    train_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/train/my_CF_processed/", limit_data=False)
    develop_dataset_info = dr.load_dataset(data_dir="/data/GIT/swde/my_data/develop/my_CF_processed/", limit_data=False)
else:
    pass

# +
# dataset, info = train_dataset_info

# import pandas as pd
# info_df = pd.DataFrame(info)
# all_text = [y for x in info_df[4] for y in x]
# all_text_no_empty = info_df[4].apply(lambda x: [y for y in x if len(y) > 0])
# all_text_no_empty = [y for x in all_text_no_empty for y in x]
# len(all_text) - len(all_text_no_empty)
# len(all_text)
# -

print(f"train_dataset_info: {len(train_dataset_info[0])}")
print(f"develop_dataset_info: {len(develop_dataset_info[0])}")

# # Train

# +
from markuplmft.fine_tuning.run_swde.markuplmodel import MarkupLModel

markup_model = MarkupLModel()
markup_model.load_pretrained_model_and_tokenizer(-1)
# -

# ### Freeze layers

if trainer_config.pop("freeze_body", False):
    for name, module in markup_model.net.named_modules():
        if name in ['token_cls', 'token_cls.dense', 'token_cls.LayerNorm', 'token_cls.decoder']:
            if name == 'token_cls':
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = True

        print(f"{name}:  {module.parameters}")
        print()

# +
# for name, module in markup_model.net.named_modules():
#     if name in ['token_cls', 'token_cls.dense', 'token_cls.LayerNorm', 'token_cls.decoder']:
#         print(f"{name}:  {module.parameters}")
#         print()

# # import torch.onnx
# # from torchviz import make_dot
# # make_dot(yhat, params=dict(list(markup_model.net.named_parameters()))).render("rnn_torchviz", format="png")
# import torch
# import torchvision

# # dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
# batch = next(iter(trainer.train_dataloader))
# # model = torchvision.models.alexnet(pretrained=True).cuda()
# model = markup_model.net

# # yhat = markup_model.net(batch.text)

# # Providing input and output names sets the display names for values
# # within the model's graph. Setting these does not change the semantics
# # of the graph; it is only for readability.
# #
# # The inputs to the network consist of the flat list of inputs (i.e.
# # the values you would pass to the forward() method) followed by the
# # flat list of parameters. You can partially specify names, i.e. provide
# # a list here shorter than the number of inputs to the model, and we will
# # only set that subset of names, starting from the beginning.
# input_names = [ "actual_input_1" ] + [ f"learned_{i}" for i in range(16) ]
# output_names = [ "output1" ]

# torch.onnx.export(model, dummy_input, "markuplm.onnx", verbose=True, input_names=input_names, output_names=output_names)

# +
# # modules = []
# for enum, x in enumerate(markup_model.net.named_modules()):
#     modules.append(x)

# # print(enum)
# # print(modules[4].)
# modules
# # for x in m:
# #     print(x)

# +
# markup_model.net.token_cls

# +
from markuplmft.fine_tuning.run_swde.trainer import Trainer

trainer = Trainer(
    model = markup_model,
    train_dataset_info = train_dataset_info,
    evaluate_dataset_info = develop_dataset_info,
    **trainer_config,
)
# -

dataset_nodes_predicted = trainer.train()
load_model_path = markup_model.save_path

# # Infer

from markuplmft.fine_tuning.run_swde.markuplmodel import MarkupLModel
print(f"load_model_path: {load_model_path}")
markup_model = MarkupLModel()
markup_model.load_trained_model(
    config_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
    tokenizer_path="/data/GIT/unilm/markuplm/markuplmft/models/my_models/",
    net_path=load_model_path,
)
markup_model

train_set_nodes_predicted = trainer.evaluate(dataset_name="train")
len(train_set_nodes_predicted)

develop_set_nodes_predicted = trainer.evaluate(dataset_name="develop")
len(develop_set_nodes_predicted)

save_path = f"develop_set_nodes_classified_epoch_{trainer_defaults['num_epochs']}.pkl"
print(save_path)
develop_set_nodes_predicted.to_pickle(save_path)

wandb.run.save()
wandb.finish()


