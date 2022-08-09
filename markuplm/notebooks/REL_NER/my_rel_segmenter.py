import pandas as pd
from REL.ner import Cmns, load_flair_ner
from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner


class RelSegmenter:
    def __init__(
        self,
        kc_wiki_csv_mapping_path="/data/GIT/unilm/markuplm/notebooks/REL_NER/kc_wiki_csv_mapping.csv",
    ):
        wiki_version = "wiki_2019"
        base_url = "/data/GIT/REL/data/generic/"
        self.mention_detector = MentionDetection(base_url, wiki_version)

        # self.tagger_ner = load_flair_ner("ner-fast")
        self.tagger_ner = load_flair_ner("ner-large")
        # self.tagger_ner = load_flair_ner("ner-ontonotes-large") #? Ontonotes dataset has more variability in its pages
        
        # self.tagger_ngram = Cmns(base_url, wiki_version, n=5)

        config = {
            "mode": "eval",
            "model_path": "ed-wiki-2019",
        }
        self.entity_disambiguator = EntityDisambiguation(base_url, wiki_version, config)

        kc_wiki_csv = pd.read_csv(kc_wiki_csv_mapping_path)

        # TODO (aimore): Uncomment this to get the wikipedia pages that google suggests
        # self.wiki_title_to_kc_mappings = dict(kc_wiki_csv.dropna(subset=['wikipedia_url']).apply(lambda x: (x['wikipedia_url'].split('/')[-1], x['taxonomy_id']), axis=1).values)

        #? By using the verified wikipedia companies, we would make sure that when returning a found company, that would be the correct one in our taxonomy. 
        #? Otherwise we could be returning companies that have nothing to do with our taxonomy despite having same name. 
        # TODO (aimore): Uncomment this to get the corrected/validated wikipedia pages
        self.wiki_title_to_kc_mappings = (
            kc_wiki_csv.dropna(subset=["title"])[["title", "taxonomy_id"]]
            .set_index("title")
            .to_dict()["taxonomy_id"]
        )

    @staticmethod
    def preprocessing(text_series):
        processed = dict()
        for e, text in text_series.items():
            processed[e] = [text, []]
        return processed

    def predict_companies(self, texts):
        input_text = self.preprocessing(texts)

        mentions_dataset, n_mentions = self.mention_detector.find_mentions(
            input_text, self.tagger_ner
        )

        predictions, timing = self.entity_disambiguator.predict(mentions_dataset)
        return predictions

    def convert_predictions_to_company_ids(self, predictions):
        wiki_titles = pd.Series(predictions.values()).apply(lambda row: row[0].get("prediction"))
        company_ids = wiki_titles.apply(lambda row: self.wiki_title_to_kc_mappings.get(row))
        return company_ids

    def get_only_org_mentions(self, mentions_dataset):
        all_org_mentions = []
        for results in mentions_dataset:
            all_org_mentions.append(
                [
                    result
                    for result in results
                    if result["tag"] == "ORG"
                ]
            )
        return all_org_mentions

    def get_mentions_dataset(self, text_series):
        input_text = self.preprocessing(text_series)

        mentions_mapping, n_mentions = self.mention_detector.find_mentions(
            input_text,
            self.tagger_ner,
        )
        return mentions_mapping

    def disambiguate(self, mentions_dataset):
        predictions, timing = self.entity_disambiguator.predict(mentions_dataset)
        return predictions


if __name__ == "__main__":
    pd.set_option(
        "display.max_columns",
        200,
        "display.max_colwidth",
        2000,
        "display.max_rows",
        200,
        "display.min_rows",
        200,
    )
    print("Run")
    rel_segmenter = RelSegmenter()

    # test_text = ["This is the COca Cola company", "Google", "Amazon", "Amazon, Google, and CocaCola"]
    # test_text = ["The first word 1 2  2  4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5  This is the COca Cola company 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 The last word ", "Google", "Amazon", "Amazon, Google, and CocaCola"]
    test_text = ["Google", "Amazon", "Amazon, Google, and CocaCola", "nasa", "Nasa", "NASA", "nasa space"]
    # test_text = ['3M', 'KT', 'EE', 'Oi', 'BT', '3M', 'EE', '3M', 'HP']

    df = pd.DataFrame(test_text)

    predictions = rel_segmenter.predict_companies(df[0])
    df["predictions"] = pd.Series(predictions.values())

    company_ids = rel_segmenter.convert_predictions_to_company_ids(predictions)
    df["company_ids"] = company_ids

    mentions = rel_segmenter.get_mentions_dataset(df[0])
    df["mentions"] = pd.Series(mentions.values())

    # org_mentions = rel_segmenter.get_only_org_mentions(df["mentions"])
    # df["org_mentions"] = org_mentions

    # predictions2 = rel_segmenter.disambiguate(df["org_mentions"])
    # df["predictions2"] = list(predictions2.values())

    print(df)