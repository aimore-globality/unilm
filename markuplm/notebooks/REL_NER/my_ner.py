import logging
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import Sequence, Tuple, List


class PastClientNER:
    def __init__(self, flair_model="flair/ner-english-large"):
        # flair_model = "flair/ner-english-fast"
        # flair_model = "flair/chunk-english-fast"
        logging.info(f"flair_model used: {flair_model}")
        self.tagger = SequenceTagger.load(flair_model)

    def predict(self, texts: Sequence[str], bs:int=32) -> List[Sentence]:
        sentences = [Sentence(text) for text in texts]
        self.tagger.predict(sentences = sentences, embedding_storage_mode="cpu", mini_batch_size = bs)
        return sentences

    def format_sentences(self, sentences: Sequence[Sentence]) -> List[Tuple[str, float, str]]:
        all_sentences = []
        for sentence in sentences:
            formatted_sentences = []
            for entity in sentence.get_spans("ner"):
                formatted_sentences.append(
                    (
                        entity.text,
                        # entity.start_position,
                        # entity.end_position,
                        entity.score,
                        entity.tag,
                    )
                )
            all_sentences.append(formatted_sentences)
        return all_sentences


if __name__ == "__main__":
    FORMAT = "[ %(asctime)s ] %(filename)20s:%(lineno)5s - %(funcName)35s() : %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    flair_ner = PastClientNER()

    sample_texts = ["There are two Past Client called BT and The Google here", "", "The company Here is very special"]

    logging.info("Predictions:")
    logging.info(f"sample_texts: {sample_texts}")
    sample_texts_predictions = flair_ner.predict(sample_texts)
    sample_texts_formated = flair_ner.format_sentences(sample_texts_predictions)
    logging.info(sample_texts_formated)
