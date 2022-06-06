import pandas as pd
import pandavro as pdx
from ast import literal_eval
from pathlib import Path
from lxml import html as lxml_html
import wandb
import os
import shutil
import os
from microcosm.api import create_object_graph
import os
import multiprocessing as mp
from pathlib import Path
import os
from pathlib import Path
from markuplmft.fine_tuning.run_swde.featurizer import Featurizer
import os


def get_annotations(annotations: pd.Series, annotation_name: str):
    return annotations.apply(
        lambda annotations: [
            annotation[annotation_name]
            for annotation in annotations
            if annotation[annotation_name]
        ]
    )


# ? Create mapping to convert gt_value_taxonomy into gt_value
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

    def __init__(self, tag="PAST_CLIENT"):
        self.tag = tag

    def load_data(self, load_data_path, limit=-1):
        print("load_data")
        self.wae_data_load_path = load_data_path
        df = pdx.read_avro(str(load_data_path / "dataset.avro"))
        df = df[:limit]
        df.annotations = df.annotations.apply(literal_eval)

        for column in ["url", "domain", "annotations"]:
            assert column in df.columns, f"Column: {column} not in DF"

        print(len(df))
        print("done")
        return df

    def format_annotation(self, df):
        print("Format_annotation...")
        df[f"{self.tag}-annotations"] = df["annotations"].apply(
            lambda annotation: annotation.get(self.tag)
        )
        df[f"{self.tag}-annotations"] = df[f"{self.tag}-annotations"].fillna("").apply(list)

        df[f"{self.tag}-gt_text"] = get_annotations(df[f"{self.tag}-annotations"], "text")
        df[f"{self.tag}-gt_value"] = get_annotations(df[f"{self.tag}-annotations"], "value")
        df[f"{self.tag}-gt_value_untax"] = df[f"{self.tag}-gt_value"].apply(
            lambda gt_values: [untaxonomize_gt_value(gt_value) for gt_value in gt_values]
        )
        df[f"{self.tag}-annotations-untax"] = df[f"{self.tag}-annotations"].apply(
            lambda annotations: [
                {
                    "gt_text": annotation["text"],
                    "gt_value_untax": untaxonomize_gt_value(annotation["value"]),
                }
                for annotation in annotations
            ]
        )
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len)

        print(f" # Annotations (gt_text): {self.count_all_labels(df)}")
        print(f" # Annotations (gt_value): {self.count_all_labels(df, 'value')}")
        return df

    def create_postive_negative_data(self, df, negative_fraction=0.1):
        print("Convert_annotated_data...")
        df_positives, df_negatives, df_negatives_sample = self.get_negative_fraction(
            df, negative_fraction
        )
        print("- Positives:")
        df_positives = self.remove_non_html_pages(
            df_positives
        )  # TODO(Aimore): Try to move this out
        print("- Negatives:")
        df_negatives_sample = self.remove_non_html_pages(
            df_negatives_sample
        )  # TODO(Aimore): Try to move this out

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
        save_intermediate_path = (
            self.wae_data_load_path
            / f"dataset_pos({len(df_positives)})_neg({len(df_negatives_sample)})_intermediate.pkl"
        )
        print(f"Saving file: {save_intermediate_path}")
        df_positives_negatives.to_pickle(save_intermediate_path)

        # ? Check the amount of annotations in each domain
        print(
            pd.DataFrame(
                df_positives_negatives.groupby("domain").sum("PAST_CLIENT-gt_text_count")
            ).sort_values("PAST_CLIENT-gt_text_count", ascending=False)
        )
        print("done")
        return df_positives_negatives

    def get_negative_fraction(self, df, negative_fraction=0.10):
        print("Get_negative_fraction...")
        df_positives = df[df[f"{self.tag}-gt_text_count"] > 0]
        df_negatives = df[df[f"{self.tag}-gt_text_count"] == 0]

        df_negatives_sample = df_negatives

        # domains_20_or_less = (
        #     df_negatives.groupby("domain")["url"]
        #     .count()[df_negatives.groupby("domain")["url"].count() <= 20]
        #     .index
        # )
        # domains_more_than_20 = (
        #     df_negatives.groupby("domain")["url"]
        #     .count()[df_negatives.groupby("domain")["url"].count() > 20]
        #     .index
        # )

        # df_negatives_sample = (
        #     df_negatives[df_negatives["domain"].isin(domains_more_than_20)]
        #     .groupby("domain")
        #     .sample(frac=negative_fraction, random_state=66)
        # )
        # df_negatives_sample = df_negatives_sample.append(
        #     df_negatives[df_negatives["domain"].isin(domains_20_or_less)]
        # )

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

        all_annotations_left = []

        for enum, row in df.iterrows():
            url = row["url"]
            if not row.isnull()[f"{self.tag}-gt_text"]:
                # clean_dom_tree = get_dom_tree(row["html"])
                # dom_tree = clean_dom_tree
                dom_tree = lxml_html.fromstring(
                    row["html"]
                )  #! The line above is slower, but that is what it is done when creating the html which the model will see

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
                        f"{len(annotations_that_cannot_be_found)} {self.tag} cannot be found in {enum } \t: {annotations_that_cannot_be_found} - {url}"
                    )
                    print()

                all_annotations_left.append(annotations_that_can_be_found)
            else:
                all_annotations_left.append(None)

        final_amount_of_label = self.count_all_labels(df)
        print(f"Final amount of labels: {final_amount_of_label}")
        print(
            f"Number of labels lost because they couldn't be found in the page: {initial_amount_of_label - final_amount_of_label}"
        )

        df[f"{self.tag}-gt_text"] = all_annotations_left
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len)
        df = df[df[f"{self.tag}-gt_text_count"] > 0]
        print("done")
        return df

    def remove_folder(self, raw_data_folder):
        print("Remove folder...")
        self.raw_data_folder = raw_data_folder
        if os.path.exists(self.raw_data_folder):
            print(f"Overwriting this folder: \n{self.raw_data_folder}")
            try:
                shutil.rmtree(self.raw_data_folder)
                print(f"REMOVED: {self.raw_data_folder}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")

        groundtruth_folder_path = self.raw_data_folder
        groundtruth_folder_path.mkdir(parents=True, exist_ok=True)

    def add_gt_counts_and_sort(self, df):
        print("Add_gt_counts_and_sort...")
        df[f"{self.tag}-gt_text_counts"] = domain_df[f"{self.tag}-gt_text"].apply(len)
        return df.sort_values(f"{self.tag}-gt_text_counts", ascending=False)

    def add_page_id(self, df):
        df["page_id"] = [str(index).zfill(4) for index in range(len(df))]
        return df

    def save_ground_truth(self, df, root_folder, domain_name):
        """
        In domain folder save a single csv file with its pages annotations
        """
        print("Save_ground_truth...")

        folder_path = root_folder / "ground_truth"
        folder_path.mkdir(parents=True, exist_ok=True)

        page_annotations_df = df[["page_id", f"{self.tag}-gt_text_counts", f"{self.tag}-gt_text"]]
        page_annotations_df.to_csv(
            folder_path / f"{domain_name}-{self.tag}.csv", sep="\t", index=False
        )

    def save_htmls(self, df, root_folder, domain_name):
        """
        In domain folder save all html pages
        """

        def save_html(html, save_path):
            Html_file = open(save_path, "w")
            Html_file.write(html)
            Html_file.close()

        print("Save htmls...")
        folder_path = root_folder / "htmls" / domain_name
        folder_path.mkdir(parents=True, exist_ok=True)
        df.apply(lambda row: save_html(row["html"], folder_path / f"{row['page_id']}.htm"), axis=1)

    def save_domain_node_features(self, df, raw_data_folder, domain_name):
        folder_path = raw_data_folder / "prepared"
        folder_path.mkdir(parents=True, exist_ok=True)
        domain_nodes = []
        for page_nodes in df["nodes"]:
            domain_nodes.extend(page_nodes)
        # TODO: Probably remove "tag" from here
        domain_nodes_df = pd.DataFrame(domain_nodes, columns=["xpath", "text", "tag", "gt_texts"])
        save_path = folder_path / f"{domain_name}.pkl"
        print(f"save_path: {save_path}")
        domain_nodes_df.to_pickle(save_path)
        return domain_nodes_df

    def save_dedup(self, domain_nodes_df, raw_data_folder, domain_name):
        folder_path = raw_data_folder / "prepared_dedup"
        folder_path.mkdir(parents=True, exist_ok=True)

        domain_nodes_df.drop_duplicates("text").to_pickle(folder_path / f"{domain_name}.pkl")

    def add_classification_label(self, nodes, gt_texts):
        """
        Node: [(xpath, text), (...)]
        gt_texts: [gt_text1, gt_text2]
        Annotated_Node: [(xpath, text, tag, [gt_text1, gt_text2]), (...)]
        """

        nodes_annotated = []
        for xpath, node_text in nodes:
            gt_text_in_node = []
            for gt_text in gt_texts:
                if f" {gt_text.strip()} ".lower() in f" {node_text.strip()} ".lower():
                    gt_text_in_node.append(gt_text)

            if len(gt_text_in_node) == 0:
                new_node_text = (xpath, node_text, None, [])
            else:
                new_node_text = (
                    xpath,
                    node_text,
                    self.tag,
                    gt_text_in_node,
                )
            nodes_annotated.append(new_node_text)
        return nodes_annotated

    def add_classification_label_to_nodes(self, df):
        df["nodes"] = df.apply(
            lambda row: self.add_classification_label(row["nodes"], row[f"{self.tag}-gt_text"]),
            axis=1,
        )
        return df


if __name__ == "__main__":
    # wandb.login()
    # self.run = wandb.init(project="LanguageModel", resume=False, tags=["convert_data"])

    dataset_name = "develop"

    # wae_data_load_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}")
    wae_data_load_path = Path(f"/data/GIT/delete/")

    raw_data_folder = Path(f"/data/GIT/delete/{dataset_name}")

    prepare_data = PrepareData(tag="PAST_CLIENT")

    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    DOC_STRIDE = 128
    MAX_SEQ_LENGTH = 384
    featurizer = Featurizer(tokenizer=tokenizer, doc_stride=DOC_STRIDE, max_length=MAX_SEQ_LENGTH)

    prepare_data.remove_folder(raw_data_folder)

    df = prepare_data.load_data(wae_data_load_path, limit=-1)  # develop size = 75824

    # df["domain"] = df["domain"].apply(lambda domain: domain.replace("-", ""))
    # assert (
    #     len(df[df["domain"].apply(lambda domain: "(" in domain or ")" in domain)]) == 0
    # )  #? Make sure domain names don't have parenthesis

    df = prepare_data.format_annotation(df)

    df_positives_negatives = prepare_data.create_postive_negative_data(df, 1)

    for domain_name, domain_df in df_positives_negatives.groupby("domain"):
        print(f"domain_name: {domain_name}")

        #! Inference
        domain_df["html"] = domain_df.apply(
            lambda row: featurizer.insert_url_into_html(row["url"], row["html"]), axis=1
        )
        domain_df["nodes"] = domain_df["html"].apply(featurizer.get_nodes)

        #! Training
        domain_df = prepare_data.add_gt_counts_and_sort(domain_df)
        domain_df = prepare_data.add_page_id(domain_df)

        prepare_data.save_ground_truth(domain_df, raw_data_folder, domain_name)
        prepare_data.save_htmls(domain_df, raw_data_folder, domain_name)

        domain_df = prepare_data.add_classification_label_to_nodes(domain_df)

        #! In this function I am sorting the columns
        domain_nodes_df = prepare_data.save_domain_node_features(
            domain_df, raw_data_folder, domain_name
        )

        # prepare_data.save_dedup(domain_nodes_df, raw_data_folder, domain_name)

        #! Inference
        domain_df["swde_features"] = domain_df.apply(
            lambda page: featurizer.get_swde_features(page["nodes"], page["url"]), axis=1
        )

        #! Training
        save_path = f"/data/GIT/delete/develop/{domain_name}.pkl"
        print(save_path)
        domain_df.to_pickle(save_path)

    # self.run.save[""]()
    # self.run.finish()
