# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.9.12 ('wae39')
#     language: python
#     name: python3
# ---

# %%
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.common.utils.parallel_tools import OptimalParallel
from web_annotation_extractor.bundles.past_client.segmentation.segmenters import PastClientSegmenter
from web_annotation_extractor.common.utils.general_utils import deserialize_annotations
import re2 as re
import pandavro as pdx
from pathlib import Path
from ast import literal_eval
import numpy as np
import pandas as pd
import unidecode
import string
from marquez.enums.annotation import EntityTag

# %%
pd.set_option("display.max_colwidth", 20)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 4)
pd.set_option("display.min_rows", 4)

graph = create_object_graph('test')

# %% [markdown]
# # Taxonomy Known companies duplicated Names

# %%
graph = create_object_graph("gazetteers")

known_company_taxonomy = []
for company in graph.known_company_taxonomy:
    if company.is_demo_company is False and company.deprecated is False:
        known_company_taxonomy.append(company)

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

# %%
# known_company_names_taxonomy.to_html("known_company_names_taxonomy.html")
known_company_names.value_counts()

# %% [markdown]
# # Analyse symbols in company names

# %%
companies_with_punc = {}
for punc in string.punctuation:
    for company in [x[0] for x in known_company_names.values]:
        if punc in company:
            company_list = companies_with_punc.get(punc, [])
            company_list.append(company)
            companies_with_punc[punc] = company_list

# %%
# pd.DataFrame.from_dict()
df = pd.DataFrame.from_dict(companies_with_punc, orient='index').T
with pd.option_context('display.max_rows', 20): 
    sorted_df = df.count().sort_values(ascending=False)
    display(sorted_df)
    display(df[sorted_df.index])


# %% [markdown]
# # Load Data

# %%
def load_data(datasets):
    all_df = pd.DataFrame()
    for dataset in datasets: #! Only Train for now
        data_path = f"/data/GIT/web-annotation-extractor/data/processed/{dataset}/dataset.avro"
        df = pdx.read_avro(data_path)    
        print(f"{dataset}: {len(df)}")
        all_df = pd.concat([all_df, df])
    print(len(all_df))
    return all_df

def get_positives(df, tag):    
    df[tag.name] = df['annotations'].apply(lambda row: row.get(tag))
    return df.dropna(subset=tag.name)


# %%
tag = EntityTag.PAST_CLIENT

overwrite = False

# datasets = ['train', 'develop', 'test']
datasets = ['train']
# datasets = ['develop']

if len(datasets) == 3: 
    dataset_name = 'all'
else:
    dataset_name = "_".join(datasets)

# %%
cached_data_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}_{tag.name}_positives.pkl")
if cached_data_path.exists() and not overwrite:
    print(f"Loaded data from: {cached_data_path}")
    positives_df = pd.read_pickle(cached_data_path)
else:
    all_df = load_data(datasets)
    all_df["annotations"] = all_df["annotations"].apply(deserialize_annotations)
    positives_df = get_positives(all_df, tag)
    print(f"Saved data at: {cached_data_path}")
    positives_df.to_pickle(cached_data_path)


# %% [markdown]
# # Breakdown Label

# %%
def explode_and_separate_annotations(df, tag_name):
    df = df.explode(tag_name)
    df["gt_text"] = df[tag_name].apply(lambda row: row.get('text'))
    df["gt_value"] = df[tag_name].apply(lambda row: row.get('value'))
    df["gt_value_untax"] = df["gt_value"].apply(lambda row: uri_to_company_name_map.get(row))
    return df


# %%
# datasets = ['train', 'develop', 'test']

# for dataset in datasets:
#     data_path = f"/data/GIT/web-annotation-extractor/data/processed/{dataset}/dataset.avro"
#     df = pdx.read_avro(data_path)
#     df["annotations"] = df["annotations"].apply(deserialize_annotations)
#     # df["annotations"] = df["annotations"].apply(literal_eval)
#     positives_df = get_positives(df, tag)
#     positives_exploded_df = explode_annotations(positives_df, tag.name)
#     # positives_exploded_df = explode_annotations(positives_df, tag)
#     positives_exploded_df = separate_annotations(positives_exploded_df, tag.name)
#     gt_value_counts = len(positives_exploded_df.dropna(subset="gt_value"))

#     print(f"Dataset: {dataset} = Pages with at least 1 annotation: {len(positives_df)} - Total number of annotations: {len(positives_exploded_df)} - Annotations with value: {gt_value_counts}")

# %%
positives_exploded_df = explode_and_separate_annotations(positives_df, tag.name)
print(f"Pages with at least one annotation: {len(positives_df)}")
print(f"Total number of gt_text: {len(positives_exploded_df)}")
print(f"Total number of gt_value: {len(positives_exploded_df.gt_value.dropna())}")
print(f"Total number of gt_value_untax: {len(positives_exploded_df.gt_value_untax.dropna())}")

# %%
pd.set_option("max_colwidth", 200)

def aggregate_per_gt_value_untax(df):
    gt_text_agg = df.groupby("gt_value_untax").agg(lambda x: sorted(list(x)))[["gt_text"]].copy()
    gt_text_agg.reset_index(inplace=True)
    gt_text_agg["regexes"] = gt_text_agg["gt_value_untax"].apply(lambda x: [x]).values
    gt_text_agg["gt_text_len"] = gt_text_agg["gt_text"].apply(len).values
    gt_text_agg = gt_text_agg.sort_values("gt_text_len")
    gt_text_agg["gt_text_no_img"] = gt_text_agg["gt_text"].apply(lambda row: [x for x in row if "http" not in x])
    gt_text_agg["gt_text_no_img_len"] = gt_text_agg["gt_text_no_img"].apply(len).values
    gt_text_agg["numb_imgs"] = gt_text_agg["gt_text_len"] - gt_text_agg["gt_text_no_img_len"]
    gt_text_agg = gt_text_agg.reset_index().drop('index',axis=1)
    return gt_text_agg


# %%
gt_text_agg = aggregate_per_gt_value_untax(positives_exploded_df.drop(['annotations', 'PAST_CLIENT'],axis=1))
print(f"Total distinct number of gt_value_untax: {len(gt_text_agg)}")

# %%
# def create_augmentation():
# ["gt_value_untax", "regexes"]

# %%
if dataset_name == "train":
    train_unique_companies = set(gt_text_agg["gt_value_untax"].unique())
if dataset_name == "develop":
    develop_unique_companies = set(gt_text_agg["gt_value_untax"].unique())


# %%
# print(f"train_unique_companies: {len(train_unique_companies)}, develop_unique_companies: {len(develop_unique_companies)}")
# print(f"intersection: {len(train_unique_companies & develop_unique_companies)}")
# print(f"difference (develop-train): {len(develop_unique_companies - train_unique_companies)}")

# %% [markdown]
# # Transform

# %%
class Transform():
    def __init__(self, transformations=["decode", "lower", "replace_ampersand", "replace_symbols", "remove_symbols", "remove_common_words", "normalize_any_space"]):
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
        common_words = [" the ", " enterprise ", " enterprises ", " group "]
        for common_word in common_words:
            return input_text.replace(common_word, " ")

    @staticmethod
    def normalize_any_space(input_text:str):
        input_text = f" {input_text.strip()} "
        input_text = re.sub('\s+', ' ', input_text)
        return input_text

        
    def transform(self, texts):        
        new_texts = []
        for text in texts:
            if text is not None: 
                for transformation in self.transform_sequence:
                    text = transformation(text)
                new_texts.append(text)
        return new_texts

transformer = Transform(["decode", "lower", "replace_ampersand", "replace_symbols", "remove_symbols", "remove_common_words", "normalize_any_space"])

# %%
gt_text_agg["regexes"] = gt_text_agg["regexes"].apply(transformer.transform)
gt_text_agg["gt_text_no_img"] = gt_text_agg["gt_text_no_img"].apply(transformer.transform)
gt_text_agg
# with pd.option_context('display.max_colwidth', 200, 'display.min_rows', 200, 'display.max_rows', 200): 
#    display(gt_text_agg)


# %% [markdown]
# # Match

# %%
from typing import Sequence
class Matcher():
    def __init__(self, method):
        compare_methods = dict(equal=self.equal, equal_token=self.equal_token, inside=self.inside)
        assert method in compare_methods.keys()
        self.method = compare_methods.get(method)

    def match(self, regex:Sequence[str], text:Sequence[str]):
        matches, not_matches, not_matches_at_all = [], [], ()
        for text_item in text:
            for regex_item in regex:
                if self.method(regex_item, text_item): #! Here for example, instead of the first item in the regex, we could have multiple
                    matches.append((regex_item, text_item))
                else:
                    not_matches.append((regex_item, text_item))
        if len(matches) == 0 and len(text) > 0: 
            not_matches_at_all = (regex, sorted(list(set(text))))
                
        return (matches, not_matches, not_matches_at_all)

    def equal(self, regex_item:str, text_item:str):
        return regex_item == text_item

    def equal_token(self, regex_item:str, text_item:str):
        return regex_item in f" {text_item} "

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
    
equal_matcher = Matcher(method="equal")
inside_matcher = Matcher(method="inside")

# %%
# print("Equal matcher:")
# gt_text_agg_matched = gt_text_agg.join(pd.DataFrame.from_records(gt_text_agg.apply(lambda row: equal_matcher.match(row["regexes"], row["gt_text_no_img"]), axis=1), columns=["matches","not_matches", "not_matches_at_all"])).copy()
# gt_text_agg_matched = count_matches(gt_text_agg_matched)
# print_count_matches(gt_text_agg_matched)

print("Inside matcher:")
gt_text_agg_matched = gt_text_agg.join(pd.DataFrame.from_records(gt_text_agg.apply(lambda row: inside_matcher.match(row["regexes"], row["gt_text_no_img"]), axis=1), columns=["matches","not_matches", "not_matches_at_all"])).copy()
gt_text_agg_matched = count_matches(gt_text_agg_matched)
print_count_matches(gt_text_agg_matched)


# %% [markdown]
# ## Get Missed matches

# %%
def get_not_matched_at_all(gt_text_agg_matched):
    return pd.DataFrame.from_records(gt_text_agg_matched["not_matches_at_all"].values, columns=["not_matches_at_all_regexes", "not_matches_at_all_texts"]).dropna().join(gt_text_agg_matched).sort_values("not_matches_at_all_len", ascending=False)[["gt_value_untax", "not_matches_at_all_texts", "not_matches_at_all_regexes"]]

not_matched_at_all = get_not_matched_at_all(gt_text_agg_matched)
not_matched_at_all_mapping = not_matched_at_all[["gt_value_untax", "not_matches_at_all_texts"]]

with pd.option_context("display.min_rows", 10, "display.max_rows", 10):
    display(not_matched_at_all)


# %% [markdown]
# # Augment CompanyNames

# %%
class NameAugmenter:
    def __init__(self, augmentations_map=None):
        if augmentations_map is None:
            self.augmentations_map = {"bt group": ["bt"]}
        else:
            self.augmentations_map = augmentations_map

    def augment_name(self, key, values):
        augmentations = self.augmentations_map.get(key, [])
        new_text = sorted(list(set(values + augmentations)))
        return new_text

def convert_not_matches_into_augmentation_map(not_matched_at_all_mapping):
    augmentations_map = not_matched_at_all_mapping.set_index("gt_value_untax").to_dict()
    return augmentations_map["not_matches_at_all_texts"]


# %%
augmentations_map = convert_not_matches_into_augmentation_map(not_matched_at_all_mapping)
augmentations_map

# %%
# name_augmenter = NameAugmenter(augmentations_map_train)
name_augmenter = NameAugmenter(augmentations_map)

# %%
gt_text_agg["new_regexes"] = gt_text_agg.apply(lambda row: (name_augmenter.augment_name(key=row["gt_value_untax"], values=row["regexes"])), axis=1)
gt_text_agg["new_regexes_len"] = gt_text_agg["new_regexes"].apply(len)
gt_text_agg.sort_values("new_regexes_len")

# %%
# print("Equal matcher:")
# gt_text_agg_matched = gt_text_agg.join(pd.DataFrame.from_records(gt_text_agg.apply(lambda row: equal_matcher.match(row["new_regexes"], row["gt_text_no_img"]), axis=1), columns=["matches","not_matches", "not_matches_at_all"])).copy()
# gt_text_agg_matched = count_matches(gt_text_agg_matched)
# print_count_matches(gt_text_agg_matched)

print("Inside matcher:")
gt_text_agg_matched = gt_text_agg.join(pd.DataFrame.from_records(gt_text_agg.apply(lambda row: inside_matcher.match(row["new_regexes"], row["gt_text_no_img"]), axis=1), columns=["matches","not_matches", "not_matches_at_all"])).copy()
gt_text_agg_matched = count_matches(gt_text_agg_matched)
print_count_matches(gt_text_agg_matched)

# %%
augmentations_map_train = augmentations_map

# %%
gt_text_agg_matched

# %%
if gt_text_agg_matched["not_matches_at_all_len"].sum() > 0:
    not_matched_at_all = get_not_matched_at_all(gt_text_agg_matched)
    not_matched_at_all_mapping = not_matched_at_all[["gt_value_untax", "not_matches_at_all_texts"]]

    with pd.option_context("display.min_rows", 10, "display.max_rows", 10):
        display(not_matched_at_all)

# %%
gt_text_agg_matched
