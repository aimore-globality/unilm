import pandas as pd
import pandavro as pdx
import re

from ast import literal_eval
from pathlib import Path
from typing import List
from lxml import html as lxml_html

import wandb

import os
import shutil
from lxml import etree

from tqdm import tqdm
from markuplmft.fine_tuning.run_swde.prepare_data import get_dom_tree
import os

from microcosm.api import create_object_graph
from __future__ import absolute_import, division, print_function

import os
import pickle

from pathlib import Path

import tqdm
from absl import app, flags
import collections


FLAGS = flags.FLAGS

flags.DEFINE_string("input_swde_path", "", "The root path to swde html page files.")
flags.DEFINE_string("output_pack_path", "", "The file path to save the packed data.")



def get_annotations(annotations: pd.Series, annotation_name: str):
    return annotations.apply(
        lambda annotations: [
            annotation[annotation_name]
            for annotation in annotations
            if annotation[annotation_name]
        ]
    )


# #? Create mapping to convert gt_value_taxonomy into gt_value
graph = create_object_graph("test")
taxonomy_to_value_mappings = dict(
    [
        (company.uri, company.name)
        for company in graph.known_company_taxonomy
    ]
)


def untaxonomize_gt_value(gt_value: str):
    gt_value_untax = taxonomy_to_value_mappings.get(gt_value)
    return gt_value_untax


class PrepareData:
    """
    - Convert CF data into SWDE format
    - Create the labels
    - Remove some data
    -
    """

    def __init__(self, tag="PAST_CLIENT", dataset_name="develop"):
        self.tag = tag
        self.dataset_name = dataset_name
        self.dataset_path = (
            f"/data/GIT/web-annotation-extractor/data/processed/{self.dataset_name}/dataset.avro"
        )

        # wandb.login()
        self.run = wandb.init(project="LanguageModel", resume=False, tags=["convert_data"])

    def load_data(self, limit=-1):
        print("load_data")
        self.df = pdx.read_avro(self.dataset_path)
        self.df = self.df[:limit]
        self.df.annotations = self.df.annotations.apply(literal_eval)

        for column in ["url", "domain", "annotations"]:
            assert column in self.df.columns, f"Column: {column} not in DF"

        #! This has to go to inference
        self.df["clean_url"] = self.df.apply(
            lambda row: self.clean_the_url(row["url"], row["domain"]), axis=1
        )

        print(len(self.df))
        print("done")

    def clean_the_url(self, string, domain):
        #! This has to go to inference
        "Remove domain and symbols from the url"
        string_without_domain = string.split(domain)[1]
        clean_string = re.sub("[%+\./:?-]", " ", string_without_domain) #? Replace symbols with spaces
        clean_string = re.sub("\s+", " ", clean_string) #? Reduce any space to one space
        return clean_string

    def convert_annotated_data(self, negative_fraction=0.1):
        print("convert_annotated_data")
        self.df["domain"] = self.df["domain"].apply(lambda domain: domain.replace("-", ""))

        assert (
            len(self.df[self.df["domain"].apply(lambda domain: "(" in domain or ")" in domain)])
            == 0
        )  # ? Make sure domain names don't have parenthesis

        self.format_annotation()

        df_positives, df_negatives, df_negatives_sample = self.get_negative_fraction(
            negative_fraction
        )
        print("- Positives:")
        df_positives = self.remove_non_html_pages(df_positives)
        print("- Negatives:")
        df_negatives_sample = self.remove_non_html_pages(df_negatives_sample)

        df_positives = self.remove_annotations_from_images(df_positives)
        df_positives = self.remove_annotations_that_cannot_be_found_on_html(df_positives)

        # ? From df_negatives_sample filter out domains that are not in df_positives
        df_negatives_sample = df_negatives_sample[
            df_negatives_sample["domain"].isin(df_positives["domain"])
        ]
        print(
            f"Positive Domains: {len(set(df_positives['domain']))} | Negative Domains: {len(set(df_negatives_sample['domain']))}"
        )
        assert (
            len(set(df_negatives_sample["domain"]) - set(df_positives["domain"])) == 0
        ), "Negatives have a domain that positive doesnt have!"

        # ? Make sure that the ratio is still the same
        df_negatives_positive_domain = df_negatives[
            df_negatives["domain"].isin(df_positives["domain"])
        ]
        final_negative_fraction = len(df_negatives_sample) / len(df_negatives_positive_domain)
        print(
            f" # of Pages (Negative Sample): {len(df_negatives_sample)} ({100*final_negative_fraction:.4f} %) \n # of Pages (Negative): {len(df_negatives_positive_domain)}"
        )
        assert negative_fraction - 0.01 < final_negative_fraction < negative_fraction + 0.01

        # ? Merge positives and negatives
        df_positives_negatives = df_positives.append(df_negatives_sample)
        print(
            f"# Total Pages (positive and negatives): {len(df_positives_negatives)} \n Total Domains: {len(set(df_positives_negatives['domain']))}"
        )

        # ? Save this dataset that is used to compare with production
        save_path = self.dataset_path.replace(
            ".avro", f"_pos({len(df_positives)})_neg({len(df_negatives_sample)})_intermediate.pkl"
        )
        print(f"Saving file: {save_path}")
        df_positives_negatives.to_pickle(save_path)

        # ? Check the amount of annotations in each domain
        pd.DataFrame(
            df_positives_negatives.groupby("domain").sum("PAST_CLIENT-gt_text_count")
        ).sort_values("PAST_CLIENT-gt_text_count", ascending=False)

        self.create_folder_with_html_and_annotations(df_positives_negatives)

        print(
            f"# of Annotations: {self.count_all_labels(df_positives_negatives)} | # of Pages: {len(df_positives_negatives)}"
        )

        print("done")
        # self.run.save()
        # self.run.finish()

    def format_annotation(self):
        self.df[f"{self.tag}-annotations"] = self.df["annotations"].apply(
            lambda annotation: annotation.get(self.tag)
        )
        self.df[f"{self.tag}-annotations"] = (
            self.df[f"{self.tag}-annotations"].fillna("").apply(list)
        )

        self.df[f"{self.tag}-gt_text"] = get_annotations(self.df[f"{self.tag}-annotations"], "text")
        self.df[f"{self.tag}-gt_value"] = get_annotations(
            self.df[f"{self.tag}-annotations"], "value"
        )
        self.df[f"{self.tag}-gt_value_untax"] = self.df[f"{self.tag}-gt_value"].apply(
            lambda gt_values: [untaxonomize_gt_value(gt_value) for gt_value in gt_values]
        )
        self.df[f"{self.tag}-annotations-untax"] = self.df[f"{self.tag}-annotations"].apply(
            lambda annotations: [
                {
                    "gt_text": annotation["text"],
                    "gt_value_untax": untaxonomize_gt_value(annotation["value"]),
                }
                for annotation in annotations
            ]
        )
        self.df[f"{self.tag}-gt_text_count"] = self.df[f"{self.tag}-gt_text"].apply(len)

        print(f" # Annotations (gt_text): {self.count_all_labels(self.df)}")
        print(f" # Annotations (gt_value): {self.count_all_labels(self.df, 'value')}")

    def get_negative_fraction(self, negative_fraction=0.10):
        df_positives = self.df[self.df[f"{self.tag}-gt_text_count"] > 0]
        df_negatives = self.df[self.df[f"{self.tag}-gt_text_count"] == 0]

        domains_20_or_less = (
            df_negatives.groupby("domain")["url"]
            .count()[df_negatives.groupby("domain")["url"].count() <= 20]
            .index
        )
        domains_more_than_20 = (
            df_negatives.groupby("domain")["url"]
            .count()[df_negatives.groupby("domain")["url"].count() > 20]
            .index
        )

        df_negatives_sample = (
            df_negatives[df_negatives["domain"].isin(domains_more_than_20)]
            .groupby("domain")
            .sample(frac=negative_fraction, random_state=66)
        )
        df_negatives_sample = df_negatives_sample.append(
            df_negatives[df_negatives["domain"].isin(domains_20_or_less)]
        )

        print(
            f"# Pages:\nNegatives: {len(df_negatives)} | Negatives sample: {len(df_negatives_sample)} | Positives:{len(df_positives)}"
        )
        return df_positives, df_negatives, df_negatives_sample

    def count_all_labels(self, df, tag_type="text"):
        return df[f"{self.tag}-gt_{tag_type}"].apply(len).sum()

    def remove_non_html_pages(self, df):
        pages_without_html_explicity = df[df["html"] == "PLACEHOLDER_HTML"]
        print(f"# of Pages that are not html explicity: {len(pages_without_html_explicity)}")
        print(
            f"# of Annotations (gt_text) that are not html explicity: {self.count_all_labels(pages_without_html_explicity)}"
        )
        df = df[df["html"] != "PLACEHOLDER_HTML"]

        def get_only_html(t):
            """Deal with XLM cases"""
            text = "NOT HTML"
            try:
                text = lxml_html.fromstring(t)
                return t
            except:
                return text

        pages_with_html = df["html"].apply(get_only_html)
        pages_without_html_implicity = df[pages_with_html == "NOT HTML"]
        print(f"# of Pages that are not html implicity: {len(pages_without_html_implicity)}")
        print(
            f"# of Annotations (gt_text) that are not html implicity: {self.count_all_labels(pages_without_html_implicity)}"
        )
        df = df[pages_with_html != "NOT HTML"]

        return df

    def remove_annotations_from_images(self, df):
        print("remove_annotations_from_images")
        print(f"# of Annotations (gt_text) before: {self.count_all_labels(df)}")
        df[f"{self.tag}-gt_text"] = df[f"{self.tag}-gt_text"].apply(
            lambda annotations: [
                annotation
                for annotation in annotations
                if "htt" not in annotation
            ]
        )
        print(f"# of Annotations (gt_text) after: {self.count_all_labels(df)}")
        print("done")
        return df

    def remove_annotations_that_cannot_be_found_on_html(self, df):
        print("remove_annotations_that_cannot_be_found_on_html")
        initial_amount_of_label = self.count_all_labels(df)

        all_new_annotations = []

        for i, row in tqdm(df.iterrows()):
            url = row["url"]
            if not row.isnull()[f"{self.tag}-gt_text"]:
                clean_dom_tree = get_dom_tree(row["html"])
                dom_tree = clean_dom_tree
                # dom_tree = lxml_html.fromstring(row['html']) #! The line above is slower, but that is what it is done when creating the html which the model will see

                annotations_that_can_be_found = []
                annotations_that_cannot_be_found = []
                for text_annotation in row[f"{self.tag}-gt_text"]:
                    found = False
                    # TODO (Aimore): This process is very similar to the one that actually annotates the nodes. It would be better if they were reused.
                    for node in dom_tree.iter():
                        if node.text:
                            if text_annotation.lower() in node.text.lower():
                                annotations_that_can_be_found.append(text_annotation)
                                found = True
                                break
                        if node.tail:
                            if text_annotation.lower() in node.tail.lower():
                                annotations_that_can_be_found.append(text_annotation)
                                found = True
                                break
                        # #? In case I want to add the images:
                        # ? 1. Don't remove img links from annotations
                        # ? 2. The img html tag contains: alt, title and src as potential places that the PC could be found.
                        # ? 3. Find a way to recreate the img node into these three pieces and incoporate then into embedding
                        # for html_tag, xpath_content in node.items():
                        #     if text_annotation in xpath_content:
                        #         annotations_that_can_be_found.append(text_annotation)
                        #         break
                    if not found:
                        annotations_that_cannot_be_found.append(text_annotation)

                if len(annotations_that_cannot_be_found) > 0:
                    print(
                        f"{len(annotations_that_cannot_be_found)} {self.tag} cannot be found in {i} \t: {annotations_that_cannot_be_found} - {url}"
                    )
                    print()

                all_new_annotations.append(annotations_that_can_be_found)
            else:
                all_new_annotations.append(None)

        df[f"{self.tag}-gt_text"] = all_new_annotations
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len)
        final_amount_of_label = self.count_all_labels(df)
        print(f"Final amount of labels: {final_amount_of_label}")
        print(
            f"Number of labels lost because they couldn't be found in the page: {initial_amount_of_label - final_amount_of_label}"
        )

        df = df[df[f"{self.tag}-gt_text_count"] > 0]
        print("done")
        return df

    def insert_url_into_html(self, clean_url, html):
        #! This has to go to inference
        element = etree.Element("title")
        element.text = f" {clean_url} "
        dom_tree = get_dom_tree(html)
        root = dom_tree.getroot()
        root.insert(0, element)
        etree.indent(dom_tree)
        etree.tostring(dom_tree, encoding=str)
        return html

    def save_html(self, save_path, html):
        Html_file = open(save_path, "w")
        Html_file.write(html)
        Html_file.close()

    def create_folder_with_html_and_annotations(self, df):
        pageid_url_mapping = {}

        raw_data_folder = Path(f"/data/GIT/swde/delete/{self.dataset_name}/")

        if os.path.exists(raw_data_folder):
            print(f"Overwriting this folder: \n{raw_data_folder}")
            try:
                shutil.rmtree(raw_data_folder)
                print(f"REMOVED: {raw_data_folder}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")

        groundtruth_folder_path = raw_data_folder / "my_CF_sourceCode" / "groundtruth"
        groundtruth_folder_path.mkdir(parents=True, exist_ok=True)

        domains = list(df.domain.value_counts().index)
        for e, domain in enumerate(domains):
            df_domain = df[df.domain == domain]

            print(f"{e:>3}: {len(df_domain):>5} page(s) - {domain:>25}")

            domain_annotations = []

            page_count = 0
            domain_len = len(df_domain)

            for enum, df_page in df_domain.iterrows():
                clean_url = df_page["clean_url"]
                html = df_page["html"]
                #! This has to go to inference
                html = self.insert_url_into_html(clean_url, html)

                raw_data_path = raw_data_folder / "WAE" / f"{domain}({domain_len})"
                raw_data_path.mkdir(parents=True, exist_ok=True)

                name = f"{str(page_count).zfill(4)}.htm"
                raw_data_path = raw_data_path / name

                pageid = f"{domain}.pickle-{name}"
                pageid_url_mapping[pageid] = df_page["url"]

                self.save_html(raw_data_path, html)
                page_count += 1

                # ? Get groundtruth for page for tag
                if not df_page.isnull()[f"{self.tag}-gt_text"]:
                    annotations = df_page[f"{self.tag}-gt_text"]
                    # ? Remove image links from text annotation / Commenting because this should have been already removed
                    # annotate = [annotation.strip() if (annotation and 'http' not in annotation.strip()) else '' for annotation in annotations]
                    annotate = [
                        annotation if annotation else ""
                        for annotation in annotations
                    ]
                else:
                    annotate = []
                domain_annotations.append(annotate)

            # ? Save groundtruth
            page_annotations_df = pd.DataFrame(domain_annotations)

            # ? Count number of annotations
            page_annotations_df["number of values"] = page_annotations_df.T.count()

            # ? Invert columns order
            cols = page_annotations_df.columns.tolist()
            page_annotations_df = page_annotations_df[cols[::-1]]

            # ? Get page index
            page_annotations_df.reset_index(inplace=True)
            page_annotations_df["index"] = page_annotations_df["index"].apply(
                lambda x: str(x).zfill(4)
            )

            # ? Add one extra row on the top
            page_annotations_df.loc[-1] = page_annotations_df.count()  # adding a row
            page_annotations_df.index = page_annotations_df.index + 1  # shifting index
            page_annotations_df = page_annotations_df.sort_index()

            groundtruth_data_tag_path = groundtruth_folder_path / f"{domain}-{self.tag}.txt"
            # groundtruth_data_tag_path = groundtruth_data_path / f"{domain}-{tag}.csv" #! Uncomment once I change to CSV

            page_annotations_df.to_csv(groundtruth_data_tag_path, sep="\t", index=False)

        # ? Save page_id to url mapping
        pd.to_pickle(
            pageid_url_mapping,
            f"/data/GIT/swde/my_data/{self.dataset_name}/my_CF_sourceCode/pageid_url_mapping.pkl",
        )


    def pack_data(swde_path="", output_pack_path="/data/GIT/swde/my_data/train/my_CF_sourceCode/wae.pickle"):
        swde_path = Path(swde_path)

        swde_data = collections.defaultdict(dict)
        print("Loading data...")

        websites_folder = os.listdir(os.path.join(swde_path))

        for website_folder in tqdm.tqdm(websites_folder, desc="Websites - Progress Bar", leave=True):
            html_filenames = os.listdir(os.path.join(swde_path, website_folder))
            html_filenames.sort()
            for html_filename in html_filenames:
                html_file_relative_path = os.path.join(website_folder, html_filename)
                print(f"Page: {html_file_relative_path}")

                html_file_absolute_path = os.path.join(swde_path, html_file_relative_path)
                with open(html_file_absolute_path) as webpage_html:
                    html_str = webpage_html.read()

                page = dict(
                    website=website_folder,  # E.g. 'capturagroup.com(8)'
                    path=html_file_relative_path,  # E.g. 'capturagroup.com(8)/0000.htm'
                    html_str=html_str,
                )

                website = website_folder.split("(")[0]
                page_id = html_filename.split(".")[0]
                swde_data[website][page_id] = page
                
        with open(output_pack_path, "wb") as output_file:
            print(f"Saving data at:  {output_file}")
            pickle.dump(swde_data, output_file)

    def clean_annotations(self):
        # ! This is not implemented yet, but I think it would be good to have a standard way of cleaning annotations
        # ! Not done yet because I need to think how that will interact with finding it in the page.
        pass


if __name__ == "__main__":
    preapre_data = PrepareData(tag="PAST_CLIENT", dataset_name="develop")
    preapre_data.load_data(limit=-1)
    preapre_data.convert_annotated_data()
    # preapre_data.pack_swde_data(
    #     # swde_path=FLAGS.input_swde_path,
    #     # pack_path=FLAGS.output_pack_path,
    # )