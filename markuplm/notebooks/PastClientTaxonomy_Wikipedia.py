# ---
# jupyter:
#   jupytext:
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
from collections import OrderedDict
import multiprocessing as mp
from tqdm import tqdm
import wikipedia

import string
from marquez.enums.annotation import EntityTag
graph = create_object_graph('test')

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
def get_information(company_name, verbose=False):
    info = OrderedDict()
    search_expression = f"{company_name} company"    
    try:
        wiki_page = wikipedia.page(search_expression)
        info["url"] = wiki_page.url
        info["title"] = wiki_page.title
        info["categories"] = wiki_page.categories
        info["summary"] = wiki_page.summary
        info["content"] = wiki_page.content
    except:
        pass
    if verbose:
        print(f"Company Name: {company_name}")
        print(f"Searched: {search_expression }")
        [print(f"{key}: {values}") for (key,values) in info.items()]
        print("-"*100)
    return info


# %%
companies = known_company_names_taxonomy['company_name']
companies_clean = [unidecode.unidecode(text) for text in companies]

# %%
all_wiki_page_info = []

parallel = True

num_cores = mp.cpu_count()
if parallel:
    with mp.Pool(num_cores) as pool, tqdm(
    total=len(companies_clean), desc="Processing data"
    ) as t:
        # for res in pool.imap_unordered(cache_page_features, all_df.values):
        for result in pool.imap(get_information, companies_clean):
            all_wiki_page_info.append(result)
            t.update()
        print(len(all_wiki_page_info))
else:
    for company_name in tqdm():
        wiki_page_info = get_information(company_name, verbose=False)
        all_wiki_page_info.append(wiki_page_info)

# %%
known_company_names_taxonomy_extended = known_company_names_taxonomy.join(pd.DataFrame(all_wiki_page_info))

# %%
pages_found_count = len(known_company_names_taxonomy_extended[["taxonomy_id", "company_name", "title", "url", "summary"]].dropna(subset='url'))
print(f"Wikipedia was able to find pages for {pages_found_count} ({100*pages_found_count/len(companies_clean):.1f} %) out of {len(companies_clean)} companies ") # 2102 

# %%
known_company_names_taxonomy_extended.to_excel("known_company_names_taxonomy_extended_by_wikipedia.xlsx")

# %%

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from pandas.io.json import json_normalize
# SPARQL query
query = '''PREFIX foaf: <http://xmlns.com/foaf/0.1/>    
PREFIX dbo: <http://dbpedia.org/ontology/>    
PREFIX dbr: <http://dbpedia.org/resource/>    
PREFIX dbp: <http://dbpedia.org/property/>    
PREFIX ling: <http://purl.org/linguistics/gold/>
SELECT DISTINCT ?a, ?dob, ?ht, ?hpn, ?g, ?name, ?c    
WHERE{{
   ?a a dbo:Athlete; 
      dbo:birthDate ?dob;
      foaf:name ?name.    
   OPTIONAL{{?a  dbo:country ?c}}    
   FILTER(LANG(?name) = "en").    
}}'''

query = '''PREFIX foaf: <http://xmlns.com/foaf/0.1/>    
PREFIX dbo: <http://dbpedia.org/ontology/>    
PREFIX dbr: <http://dbpedia.org/resource/>    
PREFIX dbp: <http://dbpedia.org/property/>    
PREFIX ling: <http://purl.org/linguistics/gold/>
SELECT DISTINCT ?a, ?dob, ?ht, ?hpn, ?g, ?name, ?c    
WHERE{{
   ?a a dbo:Athlete; 
      dbo:birthDate ?dob;
      foaf:name ?name.    
   OPTIONAL{{?a  dbo:country ?c}}    
   FILTER(LANG(?name) = "en").    
}}'''

# initialise the SPARQL endpoint
sparql = SPARQLWrapper('http://dbpedia.org/sparql')
# set query
sparql.setQuery(query)
# set the response format
sparql.setReturnFormat(JSON)
# execute the query
results = sparql.query().convert()
# normalize the json object to a pandas dataframe
df_results = json_normalize(results['results']['bindings'])


---
SELECT ?building 
WHERE {
    ?building a dbo:Building .
    <http://en.wikipedia.org/wiki/Appomattox_Court_House_National_Historical_Park> foaf:primaryTopic ?building .
} 
LIMIT 100

