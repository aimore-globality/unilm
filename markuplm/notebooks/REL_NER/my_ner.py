from flair.data import Sentence
from flair.models import SequenceTagger

class MyNER:
    def __init__(self, flair_model="flair/ner-english-large"):
        # flair_model = "flair/ner-english-fast"
        print(f"flair_model used: {flair_model}")
        self.tagger = SequenceTagger.load(flair_model)
        
    def predict(self, texts):
        sentences = [Sentence(text) for text in texts]
        self.tagger.predict(sentences)
        return sentences 

    def format_sentences(self, sentences):
        all_sentences = []
        for sentence in sentences:
            formatted_sentences = []
            for entity in sentence.get_spans('ner'):
                formatted_sentences.append(
                    (entity.text,
                    # entity.start_position,
                    # entity.end_position,
                    entity.score,
                    entity.tag,)
                )
            all_sentences.append(formatted_sentences)
        return all_sentences