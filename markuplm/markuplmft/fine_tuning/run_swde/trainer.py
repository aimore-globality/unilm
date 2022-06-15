from typing import Dict, List, Sequence
from markuplmft.fine_tuning.run_swde.featurizer import Featurizer
from markuplmft.fine_tuning.run_swde.classifier import NodeClassifier
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.eval_utils import compute_metrics_per_dataset
from torch.utils.data import DataLoader
import glob
from transformers import set_seed
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from transformers import AdamW
import transformers


class Trainer:
    def __init__(
        self,
        featurizer,
        classifier,
        train_df,
        train_batch_size: int,
        evaluate_df,
        evaluate_batch_size: int,
        num_epochs: int,
        save_model_dir: str,
        overwrite_model: bool = False,
        evaluate_during_training: bool = False,
        weight_decay: float = 0.01,
        learning_rate: float = 1e-05,
        adam_epsilon: float = 1e-8,
        warmup_ratio: float = 0,
    ):
        self.featurizer = featurizer
        self.classifier = classifier

        # ? Setting Data
        self.train_df = train_df
        self.evaluate_df = evaluate_df.copy()

        self.train_batch_size = train_batch_size
        self.evaluate_batch_size = evaluate_batch_size

        # ? Setting Training
        self.num_epochs = num_epochs
        self.evaluate_during_training = evaluate_during_training
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon

        # ? Setting Model
        self.overwrite_model = overwrite_model
        self.save_model_dir = save_model_dir
        self.training_samples = len(train_df)
        self.evaluation_samples = len(evaluate_df)

    def create_data_loader(self, features, batch_size: int, is_train: bool = True):
        return DataLoader(
            dataset=features,
            shuffle=is_train,
            batch_size=batch_size,
            # pin_memory=True,
            # num_workers=4,
        )

    def train(self):
        # wandb.login()
        # wandb.setup()

        accelerator = Accelerator(log_with="wandb")

        # defaults = {}
        # self.run = wandb.init(project="LanguageModel", config=defaults, resume=False)
        # if self.run:
        #     self.run.log(
        #         {"training_samples": self.training_samples, "evaluation_samples": self.evaluation_samples}
        #     )

        accelerator.print(f"training_pages: {self.training_samples}")
        accelerator.print(f"evaluation_pages: {self.evaluation_samples}")

        if accelerator.is_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()

        train_features = self.featurizer.feature_to_dataset(
            self.train_df["page_features"].explode().values
        )
        evaluate_features = self.featurizer.feature_to_dataset(
            self.evaluate_df["page_features"].explode().values
        )

        set_seed(66)

        train_dataloader = self.create_data_loader(
            train_features,
            self.train_batch_size,
            True,
        )
        evaluate_dataloader = self.create_data_loader(
            evaluate_features,
            self.evaluate_batch_size,
            False,
        )
        self.num_training_steps = self.num_epochs * len(train_dataloader)

        train_batches = len(train_dataloader)
        evaluate_batches = len(evaluate_dataloader)

        accelerator.print(f"Num Epochs = {self.num_epochs}")
        accelerator.print(f"Num Training steps = {self.num_training_steps}")
        accelerator.print(f"Num training data points = {len(self.train_df)}")
        accelerator.print(
            f"training_samples: {len(train_features)} | train_batches: {train_batches} | batch_size = {self.train_batch_size}"
        )
        accelerator.print(
            f"evaluate_samples: {len(evaluate_features)} | evaluate_batches: {evaluate_batches} | batch_size = {self.evaluate_batch_size}"
        )

        device = accelerator.device
        accelerator.print(f"device: {device}")

        # wandb.watch(self.classifier.model)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for param_name, param in self.classifier.model.named_parameters()
                    if not any(nd in param_name for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param
                    for param_name, param in self.classifier.model.named_parameters()
                    if any(nd in param_name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            params=optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon
        )

        accelerator.print(f"Model State - before training:")
        accelerator.print(
            f"roberta.embeddings.word_embeddings.weight:\n{self.classifier.model.state_dict()['roberta.embeddings.word_embeddings.weight']}"
        )
        accelerator.print(
            f"roberta.encoder.layer.0.attention.self.query.weight:\n{self.classifier.model.state_dict()['roberta.encoder.layer.0.attention.self.query.weight']}"
        )

        (
            self.classifier.model,
            optimizer,
            train_dataloader,
            evaluate_dataloader,
        ) = accelerator.prepare(
            self.classifier.model,
            optimizer,
            train_dataloader,
            evaluate_dataloader,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=self.num_training_steps,
        )


        #! Training step
        accelerator.print("Train...")
        progress_bar = tqdm(range(self.num_epochs), disable=not accelerator.is_main_process)
        for epoch in progress_bar:
            accelerator.print(f"Epoch: {epoch}")
            all_losses = []
            self.classifier.model.train()
            for step, batch in enumerate(train_dataloader):
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
                outputs = self.classifier.model(**inputs)
                loss = outputs[0]
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                all_losses.append(loss.item())

            accelerator.print(f"Loss: {np.mean(all_losses)}")
            progress_bar.update(1)
            if self.evaluate_during_training:
                pass  # ! Implement Evaluation function here

        #! Evaluation step
        accelerator.print("Evaluate...")
        self.classifier.model.eval()

        # accelerator.print(f"Model State - after trained:")
        # accelerator.print(f"roberta.embeddings.word_embeddings.weight:\n{self.classifier.model.state_dict()['roberta.embeddings.word_embeddings.weight']}")
        # accelerator.print(f"roberta.encoder.layer.0.attention.self.query.weight:\n{self.classifier.model.state_dict()['roberta.encoder.layer.0.attention.self.query.weight']}")

        all_logits = []
        for step, batch in enumerate(evaluate_dataloader):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
                outputs = self.classifier.model(**inputs)
            logits = outputs.logits
            all_logits.append(accelerator.gather(logits).detach().cpu())

        first_tokens_node_index = evaluate_features.relative_first_tokens_node_index
        node_probs = self.get_prob_from_first_token_of_nodes(all_logits, first_tokens_node_index)

        # TODO: Add some function to encapsulate these lines
        self.evaluate_df["html"] = self.evaluate_df["html"].astype("category")
        accelerator.print(f"Memory: {sum(self.evaluate_df.memory_usage(deep=True))/10**6:.2f} Mb")
        self.evaluate_df = self.evaluate_df.explode("nodes", ignore_index=True).reset_index()
        self.evaluate_df = self.evaluate_df.join(
            pd.DataFrame(
                self.evaluate_df.pop("nodes").tolist(),
                columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
            )
        )
        accelerator.print(
            f"Number of node prob: {len(node_probs)} | Number of nodes: {len(self.evaluate_df)}"
        )
        assert len(node_probs) == len(self.evaluate_df)
        accelerator.print(f"Memory: {sum(self.evaluate_df.memory_usage(deep=True))/10**6:.2f} Mb")
        self.evaluate_df["node_prob"] = node_probs
        self.evaluate_df["node_pred"] = node_probs > self.classifier.decision_threshold

        # TODO: move this out
        self.evaluate_df["node_gt"] = self.evaluate_df["node_gt_tag"] == "PAST_CLIENT"
        self.evaluate_df["node_pred_tag"] = self.evaluate_df["node_pred"].apply(
            lambda x: "PAST_CLIENT" if x else "none"
        )

        accelerator.print("Compute Metrics:")
        metrics_per_dataset, cm_per_dataset = compute_metrics_per_dataset(self.evaluate_df)
        accelerator.print(
            f"Node Classification Metrics per Dataset:\n {metrics_per_dataset} | cm_per_dataset: {cm_per_dataset}"
        )

        accelerator.print(f"metrics_per_dataset: {metrics_per_dataset}")
        accelerator.print(f"cm_per_dataset: {cm_per_dataset}")
        accelerator.print("... Evaluation Done")

        #! Save model
        if self.overwrite_model:
            try: 
                self.classifier.save(self.save_model_dir)
            except:
                print("COULDN'T SAVE")
            try: 
                torch.save(self.classifier.model.state_dict(), self.save_model_dir + '.pth')
            except:
                print("COULDN'T SAVE with torch")

            save_path = self.save_model_dir + "develop_df_pred.pkl"
            accelerator.print(f"Saved infered data at: {save_path}")
            self.evaluate_df.drop(["page_features"], axis=1).to_pickle(save_path)
                


    @staticmethod
    def get_prob_from_first_token_of_nodes(
        all_logits: List[float], first_tokens_node_index: List[List[int]]
    ) -> Sequence[float]:

        all_probs = torch.softmax(torch.cat(all_logits, dim=0), dim=2)[:, :, 0]
        indices = pd.Series(first_tokens_node_index).explode()
        indices = (
            indices.dropna()
        )  # TODO: There might be empty lists meaning that the window wasn't big enough to capture the beginning of any node
        node_probs = all_probs[indices.index.values, np.asarray(indices.values, int)]
        return node_probs

    def infer(self, evaluate_df):
        accelerator = Accelerator()

        if accelerator.is_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()

        evaluate_features = self.featurizer.feature_to_dataset(
            evaluate_df["page_features"].explode().values
        )

        set_seed(66)

        evaluate_dataloader = self.create_data_loader(
            evaluate_features,
            self.evaluate_batch_size,
            False,
        )

        #! Load model
        self.classifier.load(self.save_model_dir)

        (
            self.classifier.model,
            evaluate_dataloader,
        ) = accelerator.prepare(
            self.classifier.model,
            evaluate_dataloader,
        )

        device = accelerator.device
        accelerator.print(f"device: {device}")

        accelerator.print(f"Model State - loaded:")
        # accelerator.print(f"roberta.embeddings.word_embeddings.weight:\n{self.classifier.model.state_dict()['roberta.embeddings.word_embeddings.weight']}")
        # accelerator.print(f"roberta.encoder.layer.0.attention.self.query.weight:\n{self.classifier.model.state_dict()['roberta.encoder.layer.0.attention.self.query.weight']}")

        # TODO: Inference
        accelerator.print("Infer...")
        self.classifier.model.eval()
        all_logits = []
        for step, batch in enumerate(evaluate_dataloader):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    # "labels": batch[3],
                }
                outputs = self.classifier.model(**inputs)
            logits = outputs.logits
            all_logits.append(accelerator.gather(logits).detach().cpu())

        # accelerator.print(logits)

        first_tokens_node_index = evaluate_features.relative_first_tokens_node_index
        node_probs = self.get_prob_from_first_token_of_nodes(all_logits, first_tokens_node_index)

        # TODO: Add some function to encapsulate these lines
        accelerator.print(f"Memory: {sum(evaluate_df.memory_usage(deep=True))/10**6:.2f} Mb")
        evaluate_df = evaluate_df.explode("nodes", ignore_index=True).reset_index()
        evaluate_df = evaluate_df.join(
            pd.DataFrame(
                evaluate_df.pop("nodes").tolist(),
                columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
            )
        )
        accelerator.print(
            f"Number of node prob: {len(node_probs)} | Number of nodes: {len(evaluate_df)}"
        )
        assert len(node_probs) == len(evaluate_df)
        accelerator.print(f"Memory: {sum(evaluate_df.memory_usage(deep=True))/10**6:.2f} Mb")
        evaluate_df["node_prob"] = node_probs
        evaluate_df["node_pred"] = node_probs > self.classifier.decision_threshold

        # TODO: move this out
        evaluate_df["node_gt"] = evaluate_df["node_gt_tag"] == "PAST_CLIENT"
        evaluate_df["node_pred_tag"] = evaluate_df["node_pred"].apply(
            lambda x: "PAST_CLIENT" if x else "none"
        )

        accelerator.print("Compute Metrics:")
        metrics_per_dataset, cm_per_dataset = compute_metrics_per_dataset(evaluate_df)
        accelerator.print(
            f"Node Classification Metrics per Dataset:\n {metrics_per_dataset} | cm_per_dataset: {cm_per_dataset}"
        )

        accelerator.print(f"metrics_per_dataset: {metrics_per_dataset}")
        accelerator.print(f"cm_per_dataset: {cm_per_dataset}")
        accelerator.print("... Infer Done")

        #! Save infered dataset
        save_path = self.save_model_dir + "develop_df_pred.pkl"
        accelerator.print(f"Saved infered data at: {save_path}")
        evaluate_df.drop(["page_features"], axis=1).to_pickle(save_path)


if __name__ == "__main__":
    trainer_config = dict(
        dataset_to_use="all",
        train_dedup=True,  # ? Default: False
        develop_dedup=True,  # ? Default: False
        num_epochs=4,
        train_batch_size=32,  # train_batch_size 30
        evaluate_batch_size=12 * 32,
        save_model_dir="/data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/models/",
        overwrite_model=True,
    )

    if trainer_config["train_dedup"]:
        train_dedup = "_dedup"
    else:
        train_dedup = ""

    if trainer_config["develop_dedup"]:
        develop_dedup = "_dedup"
    else:
        develop_dedup = ""

    train_domains_path = glob.glob(f"/data/GIT/delete/train/processed{train_dedup}/*.pkl")
    develop_domains_path = glob.glob(f"/data/GIT/delete/develop/processed{develop_dedup}/*.pkl")

    print(f"train_domains_path: {len(train_domains_path)} - {train_domains_path[0]}")
    print(f"develop_domains_path: {len(develop_domains_path)} - {develop_domains_path[0]}")

    dataset_to_use = trainer_config["dataset_to_use"]

    # ?  Mini
    if dataset_to_use == "mini":
        train_domains_path = train_domains_path[:24]
        develop_domains_path = develop_domains_path[:8]

    # ?  All
    elif dataset_to_use == "all":
        train_domains_path = train_domains_path
        develop_domains_path = develop_domains_path

    # ?  Debug
    else:
        train_domains_path = train_domains_path[:4]
        develop_domains_path = develop_domains_path[:4]

    df_train = pd.DataFrame()
    for domain_path in train_domains_path:
        df_train = df_train.append(pd.read_pickle(domain_path))

    df_develop = pd.DataFrame()
    for domain_path in develop_domains_path:
        df_develop = df_develop.append(pd.read_pickle(domain_path))

    # featurizer_config = None
    featurizer = Featurizer()

    classifier_config = dict(decision_threshold=0.5)
    classifier = NodeClassifier(**classifier_config)

    trainer = Trainer(
        featurizer=featurizer,
        classifier=classifier,
        save_model_dir=trainer_config["save_model_dir"],
        train_df=df_train,
        train_batch_size=trainer_config["train_batch_size"],
        evaluate_df=df_develop,
        evaluate_batch_size=trainer_config["evaluate_batch_size"],
        num_epochs=trainer_config["num_epochs"],
        overwrite_model=trainer_config["overwrite_model"],
    )

    trainer.train()
    # trainer.infer(df_develop)
