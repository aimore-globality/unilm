from flair.data import Sentence
from flair.models import SequenceTagger

class MyNER:
    def __init__(self, flair_model="flair/ner-english-fast"):
        self.tagger = SequenceTagger.load(flair_model)
        # tagger = SequenceTagger.load("flair/ner-english-large")
        
    def predict(self, texts):
        sentences = [Sentence(text) for text in texts]
        self.tagger.predict(sentences)        
        return sentences 

    def format_sentences(self, sentences):        
        formatted_sentences = []
        for sentence in sentences:
            for entity in sentence.get_spans('ner'):
                formatted_sentences.append(
                    (entity.text,
                    # entity.start_position,
                    # entity.end_position,
                    entity.score,
                    entity.tag,)
                )
        return formatted_sentences