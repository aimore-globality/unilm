from collections import Counter
import logging

from pathlib import Path
import pandas as pd
from microcosm.api import create_object_graph
from typing import Sequence
from markuplmft.fine_tuning.run_swde.text_transformer import TextTransformer

graph = create_object_graph("gazetteers")

known_company_taxonomy = [
    company
    for company in graph.known_company_taxonomy
    if company.is_demo_company is False and company.deprecated is False
]

company_name_to_uri_map = dict({
    (company.name, company.uri)
    for company in known_company_taxonomy
})
uri_to_company_name_map = dict({
    (company.uri, company.name)
    for company in known_company_taxonomy
})

# known_company_names = pd.DataFrame([company.name for company in known_company_taxonomy])

# known_company_names_taxonomy = pd.DataFrame(
#     [
#         (company, company.name)
#         for company in known_company_taxonomy
#     ],
#     columns=["taxonomy_id", "company_name"],
# )

# def count_matches(gt_text_agg_matched):
#     gt_text_agg_matched["matched_len"] = gt_text_agg_matched["matches"].apply(len)
#     gt_text_agg_matched["not_matched_len"] = gt_text_agg_matched["not_matches"].apply(len)
#     gt_text_agg_matched["not_matches_at_all_len"] = gt_text_agg_matched["not_matches_at_all"].apply(
#         len
#     )
#     return gt_text_agg_matched


# def print_count_matches(gt_text_agg_matched):
#     total_unique_names = len(gt_text_agg_matched["gt_text_no_img"].explode().dropna().unique())
#     total_unique_names_not_matched = len(
#         gt_text_agg_matched["not_matches_at_all"].explode().dropna().explode().unique()
#     )
#     print(
#         f"total_unique_names_not_matched: {total_unique_names_not_matched} out of {total_unique_names}"
#     )

#     sum_all = gt_text_agg_matched[
#         ["gt_text_len", "numb_imgs", "gt_text_no_img_len", "matched_len"]
#     ].sum()
#     results = pd.DataFrame(sum_all, columns=["count"])
#     denominator = results.loc["gt_text_len"].values[0]
#     results["percent_of_total"] = results["count"].apply(lambda x: 100 * x / denominator)
#     denominator = results.loc["gt_text_no_img_len"].values[0]
#     results["percent_of_no_img"] = results["count"].apply(lambda x: 100 * x / denominator)
#     display(results)


class Segmenter:
    def __init__(self):
        self.text_transformer = TextTransformer()

        known_company_names_taxonomy = pd.DataFrame(
            [
                (company, company.name)
                for company in known_company_taxonomy
            ],
            columns=["company_id", "company_name"],
        )
        known_company_names_taxonomy["regexes"] = known_company_names_taxonomy[
            "company_name"
        ].apply(lambda company_name: [company_name])

        # ? self.companies_library = [{'company_id': None, 'company_name': None, 'regexes': []}]
        self.companies_library = known_company_names_taxonomy.to_dict("records")

    def match(self, regexes: Sequence[str], text: str) -> Sequence[str]:
        """Find in a given text a list of regexes"""
        matches = []
        for regex_item in regexes:
            if regex_item in text:
                matches.append(regex_item)
        return matches

    def find_companies(self, text: str):
        all_matches = []
        for company_id_name_regexes in self.companies_library:
            matches = self.match(company_id_name_regexes["regexes"], text)
            if matches:
                # new_items = {"company_id":[company["company_id"]]*len(company['regexes']), "matches":matches, "not_matches":not_matches}
                new_matches = {
                    "company_id": str(company_id_name_regexes["company_id"]),
                    "matches": matches,
                }
                all_matches.append(new_matches)

        return all_matches

    def number_of_companies(self) -> int:
        return len(self.companies_library)

    def number_of_regexes(self) -> int:
        return len(pd.DataFrame(self.companies_library).explode("regexes"))

    def remove_empty_regexes(self):
        companies_library = pd.DataFrame(self.companies_library)
        companies_library["regexes_len"] = companies_library["regexes"].apply(len)
        companies_library = companies_library[companies_library["regexes_len"] > 0]
        self.companies_library = companies_library.to_dict("records")

    def remove_duplicated_regexes_and_sort(self):
        companies_library = pd.DataFrame(self.companies_library)
        companies_library["regexes"] = companies_library["regexes"].apply(
            lambda row: sorted(list(set(row)))
        )
        self.companies_library = companies_library.to_dict("records")

    def transform_regexes(self):
        for company in self.companies_library:
            company["regexes"] = [
                self.text_transformer.transform(regex)
                for regex in company["regexes"]
                if self.text_transformer.transform(regex).strip() != ""
            ]

    def transform_texts(self, text: str) -> str:
        return self.text_transformer.transform(text)

    def remove_frequent_terms_with_training_metrics(
        self, domain_metrics: pd.DataFrame, precision_threshold: float = 0, save_df: bool = False
    ):
        def expand_list(l):
            return [
                y
                for x in l
                for y in x
            ]

        tp_c_df = pd.DataFrame(
            Counter(expand_list(domain_metrics["TP_seg"].values)).items(),
            columns=["regex", "tp_count"],
        ).set_index("regex")


        fp_c_df = pd.DataFrame(
            Counter(expand_list(domain_metrics["FP_seg"].values)).items(),
            columns=["regex", "fp_count"],
        ).set_index("regex")

        fn_c_df = pd.DataFrame(
            Counter([
                x[1]
                for y in domain_metrics["FN_seg"].values
                for x in y
            ]).items(),
            columns=["regex", "fn_count"],
        ).set_index("regex")
        fn_c_df.sort_values('regex', inplace=True)

        tp_fp_c_df = tp_c_df.join(fp_c_df, how="outer")

        # ? Adding 1 to the TP so that when computing the Precision of two samples,
        # ? one with a lot of FP and another with only few FP they can be differentitated by low Precision and higher Precision respectively
        
        tp_fp_c_df["tp_count"].fillna(0, inplace=True)
        tp_fp_c_df["fp_count"].fillna(0, inplace=True)
        tp_fp_c_df["tp_count"] += 1

        tp_fp_c_df["Precision"] = tp_fp_c_df.apply(
            lambda row: (row["tp_count"] / (row["tp_count"] + row["fp_count"])), axis=1
        )
        tp_fp_c_df.sort_values("Precision", inplace=True)

        if save_df:
            save_path = "Precision_score_per_regex.csv"
            logging.info(f"Saved file at:{save_path}")
            tp_fp_c_df.to_csv(save_path)
            
            print("Saving fn_c_df for FN debugging purposes.")
            save_path = "fn_c_df.csv"
            logging.info(f"Saved file at:{save_path}")
            fn_c_df.to_csv(save_path)

        regexes_to_remove = tp_fp_c_df[tp_fp_c_df["Precision"] < precision_threshold].index.values

        companies_library_df = pd.DataFrame(self.companies_library)
        companies_library_df = companies_library_df.explode("regexes")
        self.display_companies_library()

        new_companies_library = companies_library_df[
            ~companies_library_df["regexes"].isin(regexes_to_remove)
        ]

        new_companies_library = new_companies_library.groupby(["company_id", "company_name"]).agg(
            lambda regex: sorted(list(set(regex)))
        )
        new_companies_library.reset_index(inplace=True)
        self.companies_library = new_companies_library.to_dict("records")
        self.display_companies_library()

    def augment_company_names_with_training_data(self, df: pd.DataFrame):
        company_id_company_name_and_regexes = self.get_company_id_and_regexes_from_annotations(df)
        self.augment_library_with_data(company_id_company_name_and_regexes)

    def augment_company_names_with_prior_wiki_db(self):
        company_id_company_name_and_regexes = pd.read_csv(
            "extended_companies_names_based_on_wiki_DB.csv"
        ).drop("Unnamed: 0", axis=1)
        company_id_company_name_and_regexes["company_name"] = company_id_company_name_and_regexes[
            "company_id"
        ].apply(lambda x: uri_to_company_name_map.get(x))
        self.augment_library_with_data(company_id_company_name_and_regexes)

    def get_company_id_and_regexes_from_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        new_regexes = (
            df["annotations"]
            .apply(lambda row: row.get("PAST_CLIENT"))
            .dropna()
            .apply(
                lambda row: [
                    (annotation.get("value"), annotation.get("text"))
                    for annotation in row
                    if annotation.get("value") and "http" not in annotation.get("text")
                ]
            )
        )
        new_regexes_exploded = new_regexes.explode()
        new_regexes_exploded = pd.DataFrame(
            list(new_regexes_exploded.dropna()), columns=["company_id", "regexes"]
        )
        new_regexes_exploded["company_name"] = new_regexes_exploded["company_id"].apply(
            lambda company_id: uri_to_company_name_map.get(company_id)
        )
        return new_regexes_exploded  # ? new_regexes_exploded = pd.DataFrame(company_id | regexes | company_name)

    def augment_library_with_data(self, company_id_regex_name: pd.DataFrame):
        """
        - Explode all the regexes for each existing company_id in companies_library;
        - Append the new regexes
        - Group by company_id and company name
        - Remove duplicated regexes and sort them
        """
        companies_library_exploded = pd.DataFrame(self.companies_library).explode("regexes")
        new_companies_library = pd.concat([companies_library_exploded, company_id_regex_name])

        new_companies_library["company_id"] = new_companies_library["company_id"].astype(str)

        # ? Make sure there is only one regex per company
        new_companies_library = (
            new_companies_library.groupby(["company_id", "company_name"])
            .agg(lambda x: sorted(list(set(x))))
            .reset_index()
        )

        self.companies_library = new_companies_library.to_dict("records")

    def save_model(self, model_name="segmenter_trained"):
        companies_library_size = len(pd.DataFrame(self.companies_library).explode("regexes"))
        save_path = Path(f"models/segmenter/{model_name}-{companies_library_size}.pkl")
        logging.info(f"Saved the model at: {save_path}")
        if not save_path.parents[0].exists():
            logging.info(save_path.parents[0].mkdir())
        pd.to_pickle(self.companies_library, save_path)
        return save_path

    def load_model(self, model_name="segmenter_trained"):
        load_path = f"models/segmenter/{model_name}.pkl"
        logging.info(f"Loaded the model at: {load_path}")
        load_path = Path(load_path)
        self.companies_library = pd.read_pickle(load_path)

    def display_companies_library(self):
        logging.info(f"number_of_companies: {self.number_of_companies()} ")
        logging.info(f"number_of_regexes: {self.number_of_regexes()} ")


if __name__ == "__main__":
    FORMAT = "[ %(asctime)s ] %(filename)20s:%(lineno)5s - %(funcName)35s() : %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    gaz = Segmenter()
    gaz.display_companies_library()

    sample_texts = ["There are two Past Client called BT and The Google here", ""]
    matches = []

    data_path = "/data/GIT/web-annotation-extractor/data/processed/train/dataset_pos(4319)_neg(13732)_intermediate.pkl"
    df_train = pd.read_pickle(data_path)
    df_train = df_train.rename(columns={"PAST_CLIENT-annotations": "PAST_CLIENT"})
    logging.info(df_train.head(1))

    gaz.augment_company_names_with_training_data(df_train)
    gaz.display_companies_library()

    gaz.transform_regexes()
    gaz.display_companies_library()

    gaz.remove_duplicated_regexes_and_sort()
    gaz.display_companies_library()

    gaz.remove_empty_regexes()
    gaz.display_companies_library()


    logging.info("Matches:")
    for text in sample_texts:
        logging.info(f"text: {text}")

        text_t = gaz.transform_texts(text)
        logging.info(f"text_t: {text_t}")

        match = gaz.find_companies(text_t)
        logging.info(f"match ({len(match)}): {match}")
        matches.append(match)
