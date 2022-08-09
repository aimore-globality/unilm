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
#     display_name: Python 3.9.12 ('wae39_wiki')
#     language: python
#     name: python3
# ---

# %%
from REL_NER.my_rel_segmenter import RelSegmenter

# %%
from new_segmenter import Segmenter
from microcosm.api import create_object_graph
import pandas as pd
from web_annotation_extractor.evaluations.metric_functions import get_reconciliations_metrics_for_all_domains, calculate_metrics_for_dataset
import multiprocessing as mp
# from web_annotation_extractor.common.utils.general_utils import deserialize_annotations

from tqdm import tqdm
import pandavro as pdx
from pathlib import Path
from ast import literal_eval
import pandas as pd
from marquez.enums.annotation import EntityTag

# %%
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 4)
pd.set_option("display.min_rows", 4)

# %% [markdown]
# # Load Model

# %%
rel_seg = RelSegmenter()

# %% [markdown]
# # Load Data

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)
data_path = "Analyse_REL_MD_recall_dedup(284)_NOT_covered.html"
df = pd.read_html(data_path)
df = df[0].drop('Unnamed: 0', axis=1)
df

# %%
new_predictions = pd.Series(rel_seg.get_mentions_dataset(df['node_text']))
df['new_pred'] = new_predictions

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',50, 'display.min_rows',50)
save_path = data_path.replace(".html", "_Better_Model.html")
print(save_path)
# df.to_html(save_path)

# %%
df["new_mentions"] = df["new_pred"].apply(lambda row: [x['ngram'] for x in row])
correct_indices = df.apply(lambda row: True if row["node_gt_text"] in row["new_mentions"] else False, axis=1)

gt_text_in_new_rel_mentions = df[correct_indices]
gt_text_not_in_new_rel_mentions = df[~correct_indices]

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 200, 'display.max_rows',3, 'display.min_rows',3)

gt_text_not_in_new_rel_mentions

gt_text_in_new_rel_mentions

# %% [markdown]
# ## REL

# %%
# # #? Find mentions in each node
mentions = rel_seg.get_mentions_dataset(classified_df['node_text'])
classified_df["mentions"] = pd.Series(mentions)
classified_df["mentions"] = classified_df["mentions"].fillna('').apply(list)

# # #? Disambiguate each mentiions
disambiguations = rel_seg.disambiguate(classified_df["mentions"])
classified_df["disambiguations"] = pd.Series(disambiguations)
classified_df["disambiguations"] = classified_df["disambiguations"].fillna('').apply(list)

# # #? Get predictions for each disambiguation
classified_df["predictions"] = classified_df["disambiguations"].apply(lambda disambiguations: [item['prediction'] for item in disambiguations] )
# classified_df["predictions"] = classified_df["predictions"].apply(lambda disambiguations: [disambiguations[0]] if len(disambiguations) > 0 else disambiguations) #? By getting only the first item in the disambiguation there is a 3% abs variation 

# # #? Convert predictions to company_id
classified_df['rel_predictions'] = classified_df['predictions'].apply(lambda row: [rel_seg.wiki_title_to_kc_mappings.get(x) for x in row])

# # #? Convert rel_predictions to matches
classified_df['rel_matches'] = classified_df.apply(lambda row: [{'company_id': row['rel_predictions'][x], 'matches':row["mentions"][x]['ngram']} for x in range(len(row['rel_predictions']))], axis=1)

# # #? Remove empty company_id (companies that are not in our taxonomy) 
classified_df['rel_matches'] = classified_df['rel_matches'].apply(lambda row:[x for x in row if x['company_id']])

# %%
# companies_library_df = pd.DataFrame(s.companies_library)
# companies_library_df = companies_library_df.explode('regexes')
# companies_library_df
# print(f"BEFORE - Number of regexes in companies_library_df: {len(companies_library_df)}")
# regexes_to_remove = [' 123 money ', ' zvrs ']
# new_companies_library = companies_library_df[~companies_library_df['regexes'].isin(regexes_to_remove)]
# new_companies_library = new_companies_library.groupby(['company_id', 'company_name']).agg(lambda x: sorted(list(set(x))))
# new_companies_library.reset_index(inplace=True)
# new_companies_library.to_dict('records')
# # new_companies_library
# # companies_library_df["regexes"] = companies_library_df["regexes"].apply(lambda row: [x for x in row if x not in regexes_to_remove] )
# # companies_library_df = companies_library_df[companies_library_df["regexes"].apply(len) > 0 ]
# # print(f"AFTER - Number of regexes in companies_library_df: {len(companies_library_df)}")
# # self.companies_library = companies_library_df.to_dict('records')

# %%
from array import array

# rel_seg.mention_detector.wiki_db.create_index(table_name='wiki', )
# rel_seg.mention_detector.wiki_db.db.cursor

# url = ""
# local_filename = ""
# rel_seg.mention_detector.wiki_db.db.download_file(url, local_filename)

# rel_seg.mention_detector.wiki_db.lookup_wik
# word = "Arg"
# word = "ARG"
w = ["ARG"]

# word = "arg"
# rel_seg.mention_detector.wiki_db.wiki(mention=word, table_name="wiki", column_name="p_e_m")
# rel_seg.mention_detector.wiki_db.lookup_wik(w=word, table_name="wiki", column="p_e_m")
# rel_seg.mention_detector.wiki_db.lookup(word, table_name="wiki", column="emb")
DB = rel_seg.mention_detector.wiki_db.db

# c = DB.cursor()
# table_name="wiki"
# column="p_e_m"
# res = []
# for word in w:
#     e = c.execute(
#         f"select {column} from {table_name} where word = :word",
#         {"word": word},
#     ).fetchone()
#     res.append(e if e is None else array("f", e[0]).tolist())
# c.execute("COMMIT;")

# %%
import json
import sqlite3
import pandas as pd
import numpy as np
from REL_NER.my_rel_segmenter import RelSegmenter
import json

def binary_to_dict(the_binary):
    jsn = "".join(chr(int(x, 2)) for x in the_binary.split())
    d = json.loads(jsn)
    return d

def dict_to_binary(the_dict):
    # credit: https://stackoverflow.com/questions/19232011/convert-dictionary-to-bytes-and-back-again-python
    str = json.dumps(the_dict)
    binary = " ".join(format(ord(letter), "b") for letter in str)
    return binary


# %%
rel_seg = RelSegmenter()

# %%
db_path = "/data/GIT/REL/data/generic/wiki_2019/generated/entity_word_embedding.db"
db_connection = sqlite3.connect(db_path)
# df_wiki = pd.read_sql_query('select * from wiki', db_connection)
df_emb = pd.read_sql_query('select * from embeddings', db_connection)

# %%
rel_seg = RelSegmenter()

# %%
# # #? Load DB
# db_path = "/data/GIT/REL/data/generic/wiki_2019/generated/AIMORE_entity_word_embedding.db"
# db_connection = sqlite3.connect(db_path)
# new_df = pd.read_sql_query('select * from wiki', db_connection)
# print(f"Memory: {sum(new_df.memory_usage(deep=True)/10**6):.2f} Mb")

# %%
new_df["pem"] = new_df.apply(lambda x: binary_to_dict(x["p_e_m"]), axis=1)
print(f"Memory: {sum(new_df.memory_usage(deep=True)/10**6):.2f} Mb")

new_df["wikipedia_pages"] = new_df["pem"].apply(lambda row: [x[0] for x in row])
print(f"Memory: {sum(new_df.memory_usage(deep=True)/10**6):.2f} Mb")

# %%
kc_wiki_csv = pd.read_csv("/data/GIT/unilm/markuplm/notebooks/REL_NER/kc_wiki_csv_mapping.csv")
wiki_title_to_kc_mappings = (
            kc_wiki_csv.dropna(subset=["title"])[["title", "taxonomy_id"]]
            .set_index("title")
            .to_dict()["taxonomy_id"]
        )
our_taxonomy_wikipages = set(wiki_title_to_kc_mappings.keys())
# our_taxonomy_wikipages.add("#ENTITY/UNK#")

# %%
# new_df_exp = new_df.explode("wikipedia_pages")
# new_df_exp["wikipedia_pages"].value_counts()

# %%
new_df_filtered = new_df[new_df.apply(lambda row: np.any([True for x in row['wikipedia_pages'] if x in our_taxonomy_wikipages]), axis=1)]
new_df_filtered

# %%
# 

# %%
# Temp:
# new_df.to_pickle("temp_new_df.pkl")
# new_df_filtered.to_pickle("temp_new_df_filtered.pkl")
# new_df = pd.read_pickle("temp_new_df.pkl")
new_df_filtered = pd.read_pickle("temp_new_df_filtered.pkl")

# %%
# our_taxonomy_wikipages

# %%

# %%
pd.set_option('display.max_columns',200, 'display.max_colwidth', 1000, 'display.max_rows',500, 'display.min_rows',500)
new_df_filtered.sort_values('freq', ascending=False)[["word", "freq"]]

# %%
new_df_filtered[["word", "p_e_m", "lower", "freq"]]

# %% [markdown]
# # Create a new database using filtered values using Pandas

# %% [markdown]
# ## Load filtered data

# %%
db_path = "/data/GIT/REL/data/generic/wiki_2019/generated/entity_word_embedding.db"
db_connection = sqlite3.connect(db_path)
df_emb = pd.read_sql_query('select * from embeddings', db_connection)

# %%
new_df_filtered = pd.read_pickle("temp_new_df_filtered.pkl")

new_df_filtered['pem'] = new_df_filtered.apply(lambda row: [x for x in row['pem'] if x[0] in our_taxonomy_wikipages], axis=1)
new_df_filtered['p_e_m'] = new_df_filtered['pem'].apply(dict_to_binary)
new_df_filtered = new_df_filtered[["word", "p_e_m", "lower", "freq"]]

# %%
new_df_filtered

# %%
# # #? Create a new DB
conn = sqlite3.connect("/data/GIT/REL/data/generic/wiki_2019/generated/Aimore_2_entity_word_embedding.db") 
c = conn.cursor()

# %%
# # #? Create a new table
table_name = "wiki"
c.execute(f'CREATE TABLE IF NOT EXISTS {table_name} (word text primary key, p_e_m blob, lower text, freq INTEGER)')
conn.commit()

# #? Create Secondary Index
createSecondaryIndex = "CREATE INDEX if not exists idx_p_e_m ON wiki(p_e_m)"
c.execute(createSecondaryIndex)
conn.commit()
createSecondaryIndex = "CREATE INDEX if not exists idx_lower ON wiki(lower)"
c.execute(createSecondaryIndex)
conn.commit()
createSecondaryIndex = "CREATE INDEX if not exists idx_freq ON wiki(freq)"
c.execute(createSecondaryIndex)
conn.commit()

new_df_filtered.to_sql(table_name, conn, if_exists='append', index=False)

# %%
# # #? Verify
c.execute(f"SELECT * FROM {table_name}")
pd.DataFrame(c.fetchall(), columns=['word', 'p_e_m', 'lower', 'freq'])

# %%
# # #? Create a new table
table_name = "embeddings"

c.execute(f'CREATE TABLE IF NOT EXISTS {table_name} (word text primary key, emb blob)')
conn.commit()

df_emb.to_sql(table_name, conn, if_exists='append', index=False)

# %%
# #? Verify
c.execute(f"SELECT * FROM {table_name}")
pd.DataFrame(c.fetchall(), columns=['word', 'emb'])

# %%
