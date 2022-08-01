import pandas as pd
from REL.ner import Cmns, load_flair_ner
from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

class RelSegmenter:
    def __init__(self, kc_wiki_csv_mapping_path="/data/GIT/unilm/markuplm/notebooks/REL_NER/kc_wiki_csv_mapping.csv"):
        wiki_version = "wiki_2019"
        base_url = "/data/GIT/REL/data/generic/"
        self.mention_detector = MentionDetection(base_url, wiki_version)

        self.tagger_ner = load_flair_ner("ner-fast")
        # self.tagger_ngram = Cmns(base_url, wiki_version, n=5)

        config = {
            "mode": "eval",
            "model_path": "ed-wiki-2019",
        }
        self.entity_disambiguator = EntityDisambiguation(base_url, wiki_version, config)

        kc_wiki_csv = pd.read_csv(kc_wiki_csv_mapping_path)
        self.wiki_title_to_kc_mappings = kc_wiki_csv.dropna(subset=["title"])[["title", "taxonomy_id"]].set_index("title").to_dict()['taxonomy_id']
        
    def predict_companies(self, texts):
        def preprocessing(texts):
            processed = dict()
            for text in texts:
                processed[text] = [text, []]
            return processed

        input_text = preprocessing(texts)

        mentions_dataset, n_mentions = self.mention_detector.find_mentions(input_text, self.tagger_ner)

        predictions, timing = self.entity_disambiguator.predict(mentions_dataset) #? This will remove non finding entities
        if predictions:
            wiki_titles = pd.Series(predictions.values()).apply(lambda row: row[0].get('prediction'))
            company_predictions = wiki_titles.apply(lambda row: self.wiki_title_to_kc_mappings.get(row))

            return company_predictions.dropna().values
        else:
            return pd.Series([None])

    def get_only_org_mentions(self, mentions_dataset):
        all_org_mentions = []
        for results in mentions_dataset:
            all_org_mentions.append([result['mention'] for result in results if result['tag'] == 'ORG']) 
        return all_org_mentions

    def get_mentions_dataset(self, texts):
        def preprocessing(texts):
            processed = dict()
            for text in texts:
                processed[text] = [text, []]
            return processed

        input_text = preprocessing(texts)

        mentions_mapping, n_mentions = self.mention_detector.find_mentions(input_text, self.tagger_ner)
        mentions_series = texts.apply(lambda text: mentions_mapping.get(text))
        return mentions_series

    def disambiguate(self, mentions_dataset):
        predictions, timing = self.entity_disambiguator.predict(mentions_dataset) #? This will remove non finding entities
        if predictions:
            wiki_titles = pd.Series(predictions.values()).apply(lambda row: row[0].get('prediction'))
            company_predictions = wiki_titles.apply(lambda row: self.wiki_title_to_kc_mappings.get(row))

            return company_predictions
        else:
            return pd.Series([None])

if __name__ == '__main__':
    pd.set_option('display.max_columns', 200, 'display.max_colwidth', 2000, 'display.max_rows',200, 'display.min_rows',200)
    print("Run")
    rel_segmenter = RelSegmenter()

    # test_text = ["This is the COca Cola company", "Google", "Amazon"]
    test_text = ['3M', 'KT', 'EE', 'Oi', 'BT', '3M', 'EE', '3M', 'HP']
    df = pd.DataFrame(test_text)

    predictions = rel_segmenter.predict_companies(df[0])
    print(predictions)
    
    mentions = rel_segmenter.get_mentions_dataset(df[0])
    df['mentions'] = mentions
    print(df)

