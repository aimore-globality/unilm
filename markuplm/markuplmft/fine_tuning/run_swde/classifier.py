import transformers
from typing import Tuple
import numpy as np

class NodeClassifier:
    def __init__(self, config) -> None:
        self.config = config
        self.decision_threshold = self.config["decision_threshold"]
        self.model = transformers.RobertaForTokenClassification.from_pretrained('roberta-base')

    def fit(self):
        pass

    def predict_batch(self, input_data) -> Tuple[np.ndarray, np.ndarray]:
        pass
        # probability = self.model(input_data)
        # prediction = (probability >= self.decision_threshold).astype(int)
        # # self.logger.info(f"Model prediction:{prediction} | probability: {probability}")
        # return prediction, probability


    def save(self):
        pass

    def load(self):
        pass