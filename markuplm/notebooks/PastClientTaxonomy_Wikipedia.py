# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.9.12 ('wae39_wiki')
#     language: python
#     name: python3
# ---

# +
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
# -

# # Load known_company_names

# +
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
# -

# # Search and retrieve wiki page

# +
# wikipedia.search("ALDI company ", results=1)[0]
# wikipedia.page("Aldi ", auto_suggest=False)

# DisambiguationError

# +
from json import JSONDecodeError
from wikipedia import DisambiguationError

def get_information(company_name, verbose=False):
    info = OrderedDict()
    search_expression = f"{company_name} company"
    try:
        company_name_suggestion = wikipedia.search(search_expression, results=1)
        if len(company_name_suggestion) > 0:
            company_name_suggestion = company_name_suggestion[0]
        else:
            print(f"Not found {search_expression}")
            return info
    except:
        return info
    try:
        wiki_page = wikipedia.page(company_name_suggestion, auto_suggest=False)
        info["url"] = wiki_page.url
        info["title"] = wiki_page.title
        info["categories"] = wiki_page.categories
        info["summary"] = wiki_page.summary
        info["content"] = wiki_page.content

    except DisambiguationError as dis_error:
        print(f"dis_error - {search_expression} {company_name_suggestion}")
    # except JSONDecodeError as js_error:
    #     print(f"js_error - {search_expression} {company_name_suggestion}")
    except:
        return info    
        
    if verbose:
        print(f"Company Name: {company_name}")
        print(f"Searched: {company_name_suggestion}")
        [print(f"{key}: {values}") for (key,values) in info.items()]
        print("-"*100)
    return info


# -

companies = known_company_names_taxonomy['company_name']
companies_clean = [unidecode.unidecode(text) for text in companies]

# +
all_wiki_page_info = []

parallel = True
companies_clean = companies_clean[:500]
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
    for company_name in tqdm(companies_clean):
        wiki_page_info = get_information(company_name, verbose=False)
        all_wiki_page_info.append(wiki_page_info)
# -

known_company_names_taxonomy_extended = known_company_names_taxonomy.join(pd.DataFrame(all_wiki_page_info))

pages_found_count = len(known_company_names_taxonomy_extended[["taxonomy_id", "company_name", "title", "url", "summary"]].dropna(subset='url'))
print(f"Wikipedia was able to find pages for {pages_found_count} ({100*pages_found_count/len(companies_clean):.1f} %) out of {len(companies_clean)} companies ") # 2102 

# # Missed companies

# +
missed_companies = known_company_names_taxonomy_extended[known_company_names_taxonomy_extended.url.isna()].company_name.values
print(len(missed_companies))

for missed_company in missed_companies[:1]:
    print(missed_company)
    suggestion = wikipedia.search(missed_company, results=1)
    page = wikipedia.page(suggestion)
page
# -

known_company_names_taxonomy_extended

# +
# known_company_names_taxonomy_extended.to_excel("/data/GIT/unilm/markuplm/notebooks/known_company_names_taxonomy_extended_by_wikipedia.xlsx")

# +
from googleapi import google
# from google import google

num_page = 3

search_results = google.search("3M")
# -

search_results

# +
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

query = '''SELECT *
WHERE {    
    ?building foaf:isPrimaryTopicOf <http://en.wikipedia.org/wiki/Appomattox_Court_House_National_Historical_Park> .
} 
'''


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
df_results
