import torch
import transformers
from pathlib import Path
from typing import Union, Tuple
import numpy as np


class NodeClassifier:
    def __init__(self, decision_threshold:float=0.5, tag:str="PAST_CLIENT") -> None:
        self.model = transformers.RobertaForTokenClassification.from_pretrained("roberta-base")
        self.decision_threshold = decision_threshold
        self.tag = tag

    # def fit(self):
    #     pass

    # def predict_batch(self, input_data) -> Tuple[np.ndarray, np.ndarray]:
    #     pass
    #     # probability = self.model(input_data)
    #     # prediction = (probability >= self.decision_threshold).astype(int)
    #     # # self.logger.info(f"Model prediction:{prediction} | probability: {probability}")
    #     # return prediction, probability

    # def save(self, save_model_dir: Union[Path, str]):
    #     print(f"Saving Model at: {save_model_dir}")
    #     self.model.save_pretrained(save_model_dir)

    # def load(self, load_model_dir: Union[Path, str]):
    # #     print(f"Loading Model from: {load_model_dir}")
    #     self.model = transformers.RobertaForTokenClassification.from_pretrained(load_model_dir)
    #     # self.model = torch.load()(load_model_dir)
