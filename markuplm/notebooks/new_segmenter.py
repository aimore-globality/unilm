import string
from collections import Counter

from pathlib import Path
import pandas as pd
from microcosm.api import create_object_graph
import re2 as re
from html import unescape
import unidecode
from typing import Sequence

graph = create_object_graph("gazetteers")

known_company_taxonomy = [company for company in graph.known_company_taxonomy if company.is_demo_company is False and company.deprecated is False]

company_name_to_uri_map = dict({
    (company.name, company.uri)
    for company in known_company_taxonomy
})
uri_to_company_name_map = dict({
    (company.uri, company.name)
    for company in known_company_taxonomy
})

known_company_names = pd.DataFrame([company.name for company in known_company_taxonomy])
known_company_names_taxonomy = pd.DataFrame([(company, company.name) for company in known_company_taxonomy], columns=["taxonomy_id", "company_name"])
   

class Matcher:
    def __init__(self, method='inside'):
        compare_methods = dict(equal=self.equal, inside=self.inside)
        assert method in compare_methods.keys()
        self._method = compare_methods.get(method)

    def match(self, regexes:Sequence[str], text:str):
        matches = []
        for regex_item in regexes:
            if self._method(regex_item, text): #! Here for example, instead of the first item in the regex, we could have multiple
                matches.append(regex_item)
                        
        return matches


    def equal(self, regex_item:str, text_item:str):
        return regex_item == text_item

    def inside(self, regex_item:str, text_item:str):
        return regex_item in text_item

def count_matches(gt_text_agg_matched):
    gt_text_agg_matched["matched_len"] = gt_text_agg_matched["matches"].apply(len)
    gt_text_agg_matched["not_matched_len"] = gt_text_agg_matched["not_matches"].apply(len)
    gt_text_agg_matched["not_matches_at_all_len"] = gt_text_agg_matched["not_matches_at_all"].apply(len)
    return gt_text_agg_matched

def print_count_matches(gt_text_agg_matched):
    total_unique_names = len(gt_text_agg_matched['gt_text_no_img'].explode().dropna().unique())
    total_unique_names_not_matched = len(gt_text_agg_matched["not_matches_at_all"].explode().dropna().explode().unique())
    print(f"total_unique_names_not_matched: {total_unique_names_not_matched} out of {total_unique_names}")

    sum_all = gt_text_agg_matched[["gt_text_len", "numb_imgs", "gt_text_no_img_len", "matched_len"]].sum()
    results = pd.DataFrame(sum_all, columns=["count"])
    denominator = results.loc["gt_text_len"].values[0]
    results["percent_of_total"] = results["count"].apply(lambda x: 100*x/denominator)
    denominator = results.loc["gt_text_no_img_len"].values[0]
    results["percent_of_no_img"] = results["count"].apply(lambda x: 100*x/denominator)
    display(results)

matcher = Matcher(method="inside")


class Transform:
    def __init__(self, transformations=["decode", "lower", "decode_ampersand", "replace_symbols", "remove_symbols", "remove_common_words", "normalize_any_space"]):
        self.transformations = transformations
        all_transformations = dict(
            decode=self.decode,
            lower=self.lower,
            decode_ampersand=self.decode_ampersand,
            replace_symbols=self.replace_symbols,
            remove_symbols=self.remove_symbols,
            remove_common_words=self.remove_common_words,
            normalize_any_space=self.normalize_any_space,
        )
        self.transform_sequence = [all_transformations.get(transformation) for transformation in transformations]
        print(self.transform_sequence)
        
    @staticmethod
    def decode(input_text:str):
        return unidecode.unidecode(input_text)

    @staticmethod
    def lower(input_text:str):
        return input_text.lower()
    
    @staticmethod
    def decode_ampersand(input_text:str):
        return unescape(input_text)
        # input_text = input_text.replace("amp&;", '&')
        # return input_text.replace('amp&', '&')

    @staticmethod
    def replace_symbols(input_text:str, symbol_to_replace=' '):
        all_symbols = set(string.punctuation) - {'&'}
        for symbol in all_symbols:
            input_text = input_text.replace(symbol, symbol_to_replace)
        return input_text

    @staticmethod
    def remove_symbols(input_text:str): 
        all_symbols = set(string.punctuation) - {'&'}
        for symbol in all_symbols:
            input_text = input_text.replace(symbol, '')
        return input_text

    @staticmethod
    def remove_common_words(input_text:str):
        input_text = f" {input_text.strip()} "
        common_words = ["the", "enterprises", "group", "plc", "company", "limited"]
        for common_word in common_words:
            input_text = input_text.replace(f" {common_word} ", " ")
        return input_text

    @staticmethod
    def normalize_any_space(input_text:str):
        input_text = re.sub('\s+', ' ', input_text)
        input_text = f" {input_text.strip()} "
        return input_text

    def transform(self, text):
        if text is not None: 
            for transformation in self.transform_sequence:
                text = transformation(text)
            return text


class Segmenter:
    def __init__(self):
        self.transformer = Transform()
        self.matcher = Matcher()

        known_company_names_taxonomy = pd.DataFrame([(company, company.name) for company in known_company_taxonomy], columns=["company_id", "company_name"])
        known_company_names_taxonomy['regexes'] = known_company_names_taxonomy['company_name'].apply(lambda x:[x])

        #? self.companies_library = [{'company_id': None, 'company_name': None, 'regexes': []}]
        self.companies_library = known_company_names_taxonomy.to_dict('records')

    def number_of_companies(self):
        return len(self.companies_library)

    def number_of_regexes(self):
        return len(pd.DataFrame(self.companies_library).explode('regexes'))

    def find_companies(self, text:str):
        all_matches = []
        for company_id_name_regexes in self.companies_library:
            matches = self.matcher.match(company_id_name_regexes['regexes'], text)
            if matches:
                # new_items = {"company_id":[company["company_id"]]*len(company['regexes']), "matches":matches, "not_matches":not_matches}
                new_matches = {"company_id": str(company_id_name_regexes['company_id']), "matches": matches}
                all_matches.append(new_matches)
        
        return all_matches

    def remove_duplicated_regexes_and_sort(self):
        companies_library = pd.DataFrame(self.companies_library)
        companies_library["regexes"] = companies_library["regexes"].apply(lambda row: sorted(list(set(row))))
        self.companies_library = companies_library.to_dict('records')

    def transform_regexes(self):
        for enum, company in enumerate(self.companies_library):
            self.companies_library[enum]["regexes"] = [self.transformer.transform(x) for x in self.companies_library[enum]["regexes"] if self.transformer.transform(x).strip() != '']            

    def transform_texts(self, texts):
        return self.transformer.transform(texts)

    def remove_frequent_terms_with_training_metrics(self, domain_metrics:pd.DataFrame, f1_min_percentage = 0.2, save_df=False):
        tp_c_df = pd.DataFrame(Counter([x for y in domain_metrics["TP_seg"].values for x in y]).items(), columns=["regex", "tp_count"]).set_index("regex")
        fp_c_df = pd.DataFrame(Counter([x for y in domain_metrics["FP_seg"].values for x in y]).items(), columns=["regex", "fp_count"]).set_index("regex")
        fn_c_df = pd.DataFrame(Counter([x[1] for y in domain_metrics["FN_seg"].values for x in y]).items(), columns=["regex", "fn_count"]).set_index("regex")

        tp_fp_c_df = tp_c_df.join(fp_c_df)
        tp_fp_fn_c_df = tp_fp_c_df.join(fn_c_df)
        tp_fp_fn_c_df = tp_fp_fn_c_df.fillna(0)

        tp_fp_fn_c_df["Precision"] = tp_fp_fn_c_df.apply(lambda row: (row['tp_count']/(row['tp_count']+row['fp_count'])), axis=1)
        tp_fp_fn_c_df["Recall"] = tp_fp_fn_c_df.apply(lambda row: (row['tp_count']/(row['tp_count']+row['fn_count'])), axis=1)
        tp_fp_fn_c_df["F1"] = tp_fp_fn_c_df.apply(lambda row: 2*(row['Precision']*row["Recall"]) / (row['Precision']+row["Recall"]) , axis=1)

        tp_fp_fn_c_df.sort_values("F1", inplace=True)
        if save_df:
            save_path = "f1_score_per_regex.csv"
            print(f"Saved file at:{save_path}")
            tp_fp_fn_c_df.to_csv(save_path)
        
        regexes_to_remove = tp_fp_fn_c_df[tp_fp_fn_c_df["F1"] < f1_min_percentage].index.values

        companies_library_df = pd.DataFrame(self.companies_library)
        companies_library_df = companies_library_df.explode('regexes')
        print(f"BEFORE - Number of companies = {self.number_of_companies()} and regexes = {self.number_of_regexes()} in companies_library_df.")

        new_companies_library = companies_library_df[~companies_library_df['regexes'].isin(regexes_to_remove)]

        new_companies_library = new_companies_library.groupby(['company_id', 'company_name']).agg(lambda x: sorted(list(set(x))))
        new_companies_library.reset_index(inplace=True)
        self.companies_library = new_companies_library.to_dict('records')
        print(f"AFTER - Number of companies = {self.number_of_companies()} and regexes = {self.number_of_regexes()} in companies_library_df.")


    def augment_company_names_with_training_data(self, df:pd.DataFrame):
        company_id_company_name_and_regexes = self.get_company_id_and_regexes_from_annotations(df)
        self.augment_library_with_data(company_id_company_name_and_regexes)

    def augment_company_names_with_prior_wiki_db(self):
        company_id_company_name_and_regexes = pd.read_csv("extended_companies_names_based_on_wiki_DB.csv").drop("Unnamed: 0", axis=1)
        company_id_company_name_and_regexes["company_name"] = company_id_company_name_and_regexes["company_id"].apply(lambda x:uri_to_company_name_map.get(x))
        self.augment_library_with_data(company_id_company_name_and_regexes)

    def get_company_id_and_regexes_from_annotations(self, df:pd.DataFrame) -> pd.DataFrame:
        new_regexes = df["annotations"].apply(lambda row: row.get("PAST_CLIENT") ).dropna().apply(lambda row: [(x.get('value'), x.get('text')) for x in row if x.get('value') and "http" not in x.get('text')])
        new_regexes = new_regexes.dropna().explode()
        new_regexes = pd.DataFrame(list(new_regexes.dropna()), columns=["company_id", "regexes"])
        new_regexes["company_name"] = new_regexes["company_id"].apply(lambda x:uri_to_company_name_map.get(x))
        return new_regexes #? company_id: [regexes]

    def augment_library_with_data(self, company_id_regexes:pd.DataFrame):
        """
        input: pandas df exploded (one row with possibly company_id duplicated and multiple regexes) with two columns = ["company_id", "company_name", "regexes"] 
        """
        companies_library_exploded = pd.DataFrame(self.companies_library).explode("regexes")
        new_companies_library = pd.concat([companies_library_exploded, company_id_regexes])

        new_companies_library["company_id"] = new_companies_library["company_id"].astype(str)

        #? Make sure we only have one instance of the same regex per company
        new_companies_library = new_companies_library.groupby(['company_id', 'company_name']).agg(lambda x: sorted(list(set(x))))

        new_companies_library.reset_index(inplace=True)
        new_companies_library["regexes"] = new_companies_library["regexes"].apply(lambda x: sorted(list(set(x))))
        self.companies_library = new_companies_library.to_dict('records')

    def save_model(self, model_name="segmenter_trained"):
        companies_library_size = len(pd.DataFrame(self.companies_library).explode('regexes'))
        save_path = f"models/segmenter/{model_name}-{companies_library_size}.pkl"
        print(f"Saved the model at: {save_path}")
        save_path = Path(save_path)
        if not save_path.parents[0].exists():
            print(save_path.parents[0].mkdir())
        pd.to_pickle(self.companies_library, save_path)
        return save_path

    def load_model(self, model_name="segmenter_trained"):
        load_path = f"models/segmenter/{model_name}.pkl"
        print(f"Loaded the model at: {load_path}")
        load_path = Path(load_path)
        self.companies_library = pd.read_pickle(load_path)