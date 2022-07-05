import string
import pandas as pd
from microcosm.api import create_object_graph
import re2 as re

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

    # def match(self, regexes:Sequence[str], text:Sequence[str]):
    #     matches, not_matches, not_matches_at_all = [], [], ()
    #     for text_item in text:
    #         for regex_item in regexes:
    #             if self._method(regex_item, text_item): #! Here for example, instead of the first item in the regex, we could have multiple
    #                 matches.append((regex_item, text_item))
    #             else:
    #                 not_matches.append((regex_item, text_item))
    #     if len(matches) == 0 and len(text) > 0: 
    #         not_matches_at_all = (regexes, sorted(list(set(text))))
                
    #     return (matches, not_matches, not_matches_at_all)
    
    # def match(self, regexes:Sequence[str], text:str):
    #     matches, not_matches = [], []
    #     for regex_item in regexes:
    #         if self._method(regex_item, text): #! Here for example, instead of the first item in the regex, we could have multiple
    #             matches.append((regex_item, text))
    #         else:
    #             not_matches.append((regex_item, text))
                        
    #     return (matches, not_matches)

    def match(self, regexes:Sequence[str], text:str):
        matches, not_matches = [], []
        for regex_item in regexes:
            if self._method(regex_item, text): #! Here for example, instead of the first item in the regex, we could have multiple
                matches.append(regex_item)
            else:
                not_matches.append(regex_item)
                        
        return (matches, not_matches)


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
    def __init__(self, transformations=["decode", "lower", "replace_ampersand", "replace_symbols", "remove_symbols", "remove_common_words", "normalize_any_space"]):
        self.transformations = transformations
        all_transformations = dict(
            decode=self.decode,
            lower=self.lower,
            replace_ampersand=self.replace_ampersand,
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
    def replace_ampersand(input_text:str):
        return input_text.replace('amp&', '&')

    @staticmethod
    def replace_symbols(input_text:str, symbol_to_replace=' '):
        all_symbols = set(string.punctuation)
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
        common_words = ["the", "enterprise", "enterprises", "group", "plc", "company", "limited"]
        for common_word in common_words:
            input_text = input_text.replace(f" {common_word} ", " ")
        return input_text

    @staticmethod
    def normalize_any_space(input_text:str):
        input_text = re.sub('\s+', ' ', input_text)
        input_text = f" {input_text.strip()} "
        return input_text

        
    # def transform(self, texts):        
    #     new_texts = []
    #         if text is not None: 
    #             for transformation in self.transform_sequence:
    #                 text = transformation(text)
    #             new_texts.append(text)
    #     return new_texts

    def transform(self, text):
        if text is not None: 
            for transformation in self.transform_sequence:
                text = transformation(text)
            return text

# transformer = Transform(["decode", "lower", "replace_ampersand", "replace_symbols", "remove_symbols", "remove_common_words", "normalize_any_space"])
# transformer = Transform(["decode", "lower"])
# transformer = Transform(["decode"])

class Segmenter:
    def __init__(self):
        self.transformer = Transform()
        self.matcher = Matcher()

        known_company_names_taxonomy = pd.DataFrame([(company, company.name) for company in known_company_taxonomy], columns=["company_id", "company_name"])
        known_company_names_taxonomy['regexes'] = known_company_names_taxonomy['company_name'].apply(lambda x:[x])

        #? self.companies_library = [{'company_id': None, 'company_name': None, 'regexes': []}]
        self.companies_library = known_company_names_taxonomy.to_dict('records')

    def find_companies(self, text:str):
        text_results = []
        for company in self.companies_library:
            matches, not_matches = self.matcher.match(company['regexes'], text)
            if matches:              
                new_items = {"company_id":[company["company_id"]]*len(company['regexes']), "matches":matches, "not_matches":not_matches}
                text_results.append(new_items)
            # else:
                # not_matches_at_all = (regexes, sorted(list(set(text))))

        
        return text_results

    def transform_regexes(self):
        for enum, company in enumerate(self.companies_library):
            self.companies_library[enum]["regexes"] = [self.transformer.transform(x) for x in self.companies_library[enum]["regexes"]]

    def transform_texts(self, texts):
        return self.transformer.transform(texts)

    def augment_library_with_training_data(self, company_id_company_name_regexes:pd.DataFrame):
        companies_library_exploded = pd.DataFrame(self.companies_library).explode("regexes")
        new_companies_library = pd.concat([companies_library_exploded, company_id_company_name_regexes])

        new_companies_library["company_id"] = new_companies_library["company_id"].astype(str)

        #? Make sure we only have one instance of the same regex per company
        new_companies_library = new_companies_library.groupby(['company_id', 'company_name']).agg(lambda x: sorted(list(set(x))))

        new_companies_library.reset_index(inplace=True)
        self.companies_library = new_companies_library
