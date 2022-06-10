import logging
from typing import Sequence
import pandas as pd
from tqdm import tqdm
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


def get_annotations(annotations: pd.Series, annotation_name: str) -> pd.Series:
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


def untaxonomize_gt_value(gt_value: str) -> str:
    gt_value_untax = taxonomy_to_value_mappings.get(gt_value)
    return gt_value_untax


class PrepareData:
    """
    - Convert CF data into SWDE format
    - Create the labels
    - Remove some data
    -
    """

    def __init__(self, tag: str = "PAST_CLIENT"):
        self.tag = tag

    def load_data(self, load_data_path: str, limit: int = -1) -> pd.DataFrame:
        logging.info("Load_data...")
        self.wae_data_load_path = load_data_path
        df = pdx.read_avro(str(load_data_path / "dataset.avro"))
        df = df[:limit]
        df.annotations = df.annotations.apply(literal_eval)

        for column in ["url", "domain", "annotations"]:
            assert column in df.columns, f"Column: {column} not in DF"

        logging.info(len(df))
        return df

    def format_annotation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Format_annotation...")
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

        logging.info(f" # Annotations (gt_text): {self.count_all_labels(df)}")
        logging.info(f" # Annotations (gt_value): {self.count_all_labels(df, 'value')}")
        return df

    def create_postive_negative_data(
        self,
        df: pd.DataFrame,
        negative_fraction: float,
    ) -> pd.DataFrame:
        logging.info("Convert_annotated_data...")
        df_positives, df_negatives, df_negatives_sample = self.get_negative_fraction(
            df, negative_fraction
        )
        logging.info("- Positives:")
        # TODO(Aimore): Try to move this out (this also has to happen during inference)
        df_positives = self.remove_non_html_pages(df_positives)
        logging.info("- Negatives:")
        df_negatives_sample = self.remove_non_html_pages(df_negatives_sample)

        df_positives = self.remove_annotations_from_images(df_positives)
        df_positives = self.remove_annotations_that_cannot_be_found_on_html(df_positives)

        # ? From df_negatives_sample filter out domains that are not in df_positives
        df_negatives_sample = df_negatives_sample[
            df_negatives_sample["domain"].isin(df_positives["domain"])
        ]
        positive_domains = set(df_positives["domain"])
        negative_domains = set(df_negatives_sample["domain"])
        logging.info(f"Positive Domains: {len(positive_domains)}")
        logging.info(f"Negative Domains: {len(negative_domains)}")
        df_negatives_sample = df_negatives_sample[
            df_negatives_sample["domain"].isin(positive_domains)
        ]
        assert (
            len(negative_domains - positive_domains) == 0
        ), "Negatives have domains that Positives don't have!"  #! Uncomment

        # ? Make sure that the ratio is still the same
        df_negatives_positive_domain = df_negatives[
            df_negatives["domain"].isin(df_positives["domain"])
        ]
        final_negative_fraction = len(df_negatives_sample) / len(df_negatives_positive_domain)
        logging.info(
            f" # of Pages (Negative Sample): {len(df_negatives_sample)} ({100*final_negative_fraction:.4f} %) \n # of Pages (Negative): {len(df_negatives_positive_domain)}"
        )
        assert (
            negative_fraction - 0.01 < final_negative_fraction < negative_fraction + 0.01
        ), f"Not in the range ({negative_fraction - 0.01}, {negative_fraction - 0.01}) final_negative_fraction: {final_negative_fraction}"  #! Uncomment

        # ? Merge positives and negatives
        df_positives_negatives = df_positives.append(df_negatives_sample)
        logging.info(
            f"# Total Pages (positive and negatives): {len(df_positives_negatives)} \n Total Domains: {len(set(df_positives_negatives['domain']))}"
        )

        # ? Save this dataset that is used to compare with production
        save_intermediate_path = (
            self.wae_data_load_path
            / f"dataset_pos({len(df_positives)})_neg({len(df_negatives_sample)})_intermediate.pkl"
        )
        logging.info(f"Saving file: {save_intermediate_path}")
        df_positives_negatives.to_pickle(save_intermediate_path)

        # ? Check the amount of annotations in each domain
        logging.info(
            pd.DataFrame(
                df_positives_negatives.groupby("domain").sum("PAST_CLIENT-gt_text_count")
            ).sort_values("PAST_CLIENT-gt_text_count", ascending=False)
        )
        logging.info("done")
        return df_positives_negatives

    def get_negative_fraction(
        self,
        df: pd.DataFrame,
        negative_fraction: float,
    ) -> Sequence[pd.DataFrame]:
        logging.info("Get_negative_fraction...")
        df_positives = df[df[f"{self.tag}-gt_text_count"] > 0]
        df_negatives = df[df[f"{self.tag}-gt_text_count"] == 0]

        df_negatives_sample = df_negatives

        if negative_fraction != 1:
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

        logging.info(
            f"# Pages: Negatives: {len(df_negatives)} | Negatives sample: {len(df_negatives_sample)} | Positives:{len(df_positives)}"
        )
        return df_positives, df_negatives, df_negatives_sample

    def count_all_labels(self, df, tag_type="text"):
        return df[f"{self.tag}-gt_{tag_type}"].apply(len).sum()

    def remove_non_html_pages(self, df: pd.DataFrame) -> pd.DataFrame:
        pages_without_html_explicity = df[df["html"] == "PLACEHOLDER_HTML"]
        logging.info(f"# of Pages that are not html explicity: {len(pages_without_html_explicity)}")
        logging.info(
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
        logging.info(f"# of Pages that are not html implicity: {len(pages_without_html_implicity)}")
        logging.info(
            f"# of Annotations (gt_text) that are not html implicity: {self.count_all_labels(pages_without_html_implicity)}"
        )
        df = df[pages_with_html != "NOT HTML"]

        return df

    def remove_annotations_from_images(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("remove_annotations_from_images")
        logging.info(f"# of Annotations (gt_text) before: {self.count_all_labels(df)}")
        df[f"{self.tag}-gt_text"] = df[f"{self.tag}-gt_text"].apply(
            lambda annotations: [
                annotation
                for annotation in annotations
                if "htt" not in annotation
            ]
        )
        logging.info(f"# of Annotations (gt_text) after: {self.count_all_labels(df)}")
        logging.info("done")
        return df

    def remove_annotations_that_cannot_be_found_on_html(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("remove_annotations_that_cannot_be_found_on_html")
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
                        # ? 2. The img html node_gt_tag contains: alt, title and src as potential places that the PC could be found.
                        # ? 3. Find a way to recreate the img node into these three pieces and incoporate then into embedding
                        # for html_tag, xpath_content in node.items():
                        #     if text_annotation in xpath_content:
                        #         annotations_that_can_be_found.append(text_annotation)
                        #         break
                    if not found:
                        annotations_that_cannot_be_found.append(text_annotation)

                if len(annotations_that_cannot_be_found) > 0:
                    logging.info(
                        f"{len(annotations_that_cannot_be_found)} {self.tag} cannot be found in {enum } \t: {annotations_that_cannot_be_found} - {url}"
                    )
                all_annotations_left.append(annotations_that_can_be_found)
            else:
                all_annotations_left.append(None)

        final_amount_of_label = self.count_all_labels(df)
        logging.info(f"Final amount of labels: {final_amount_of_label}")
        logging.info(
            f"Number of labels lost because they couldn't be found in the page: {initial_amount_of_label - final_amount_of_label}"
        )

        df[f"{self.tag}-gt_text"] = all_annotations_left
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len)
        df = df[df[f"{self.tag}-gt_text_count"] > 0]
        logging.info("done")
        return df

    def remove_folder(self, raw_data_folder: str):
        logging.info("Remove folder...")
        self.raw_data_folder = raw_data_folder
        if os.path.exists(self.raw_data_folder):
            logging.info(f"Overwriting this folder: \n{self.raw_data_folder}")
            try:
                shutil.rmtree(self.raw_data_folder)
                logging.info(f"REMOVED: {self.raw_data_folder}")
            except OSError as e:
                logging.info(f"Error: {e.filename} - {e.strerror}.")

        groundtruth_folder_path = self.raw_data_folder
        groundtruth_folder_path.mkdir(parents=True, exist_ok=True)

    def add_gt_counts_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates tag-gt_text_count column.
        """
        logging.debug("Add_gt_counts_and_sort...")
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len)
        return df.sort_values(f"{self.tag}-gt_text_count", ascending=False)

    def add_page_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates page_id column.
        """
        df["page_id"] = [str(index).zfill(4) for index in range(len(df))]
        return df

    def save_ground_truth(self, df, domain_name, root_folder):
        """
        In domain folder save a single csv file with its pages annotations
        """
        logging.debug("Save_ground_truth...")

        folder_path = root_folder / "ground_truth"
        folder_path.mkdir(parents=True, exist_ok=True)

        page_annotations_df = df[["page_id", f"{self.tag}-gt_text_count", f"{self.tag}-gt_text"]]
        page_annotations_df.to_csv(
            folder_path / f"{domain_name}-{self.tag}.csv", sep="\t", index=False
        )

    def save_htmls(self, df: pd.DataFrame, domain_name: str, root_folder: Path):
        """
        In domain folder save all html pages
        """

        def save_html(html, save_path):
            Html_file = open(save_path, "w")
            Html_file.write(html)
            Html_file.close()

        logging.debug("Save htmls...")
        folder_path = root_folder / "htmls" / domain_name
        folder_path.mkdir(parents=True, exist_ok=True)
        df.apply(lambda row: save_html(row["html"], folder_path / f"{row['page_id']}.htm"), axis=1)

    # def save_domain_node_features(self, df, raw_data_folder, domain_name):
    #     folder_path = raw_data_folder / "prepared"
    #     folder_path.mkdir(parents=True, exist_ok=True)
    #     domain_nodes = []
    #     for page_nodes in df["nodes"]:
    #         domain_nodes.extend(page_nodes)
    #     domain_nodes_df = pd.DataFrame(
    #         domain_nodes, columns=["xpath", "text", "node_gt_tag", "node_gt_text"]
    #     )
    #     save_path = folder_path / f"{domain_name}.pkl"
    #     logging.info(f"save_path: {save_path}")
    #     domain_nodes_df.to_pickle(save_path)
    #     return domain_nodes_df

    def create_dedup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly remove duplicated nodes across the entire domain
        Note: Some urls might have been droped due to the deduplication
        """
        # ? Expand nodes
        df_nodes = df.explode("nodes").reset_index()
        # ? Join expanded nodes into df
        df_nodes = df_nodes.join(
            pd.DataFrame(
                df_nodes.pop("nodes").tolist(),
                columns=["xpath", "node_text", "node_gt_tag", "node_gt_text"],
            )
        )
        # ? Remove duplicates
        df_nodes_dedup = df_nodes.drop_duplicates("node_text")
        # ? Group dedup nodes by page
        ddd = (
            df_nodes_dedup[["url", "xpath", "node_text", "node_gt_tag", "node_gt_text"]]
            .groupby("url")
            .agg(lambda x: list(x))
            .reset_index()
        )
        # ? Combine dedup nodes into a single column (as they were before )
        ddd["nodes"] = ddd.apply(
            lambda x: list(zip(x["xpath"], x["node_text"], x["node_gt_tag"], x["node_gt_text"])),
            axis=1,
        )
        # ? Remove pages that don't contain nodes anymore due to dedup
        ddd = ddd.dropna(subset=["nodes"])
        # ? Remove columns that are not relevant anymore
        ddd = ddd.drop(["xpath", "node_text", "node_gt_tag", "node_gt_text"], axis=1)
        # ? Merge both datasets (size of ddd might be different from the original df)
        ddd = ddd.set_index("url")
        df = df.set_index("url")
        df = df.drop(["nodes"], axis=1)
        df = ddd.join(df)
        assert len(df) > 0
        return df

    def save_dedup(
        self,
        df: pd.DataFrame,
        domain_name: str,
        raw_data_folder: Path,
    ):
        folder_path = raw_data_folder / "dedup"
        folder_path.mkdir(parents=True, exist_ok=True)
        save_path = folder_path / f"{domain_name}.pkl"
        logging.debug(f"Saved dedup file at: {save_path} ({len(df)})")
        df.to_pickle(save_path)

    def add_classification_label(
        self,
        nodes: Sequence[str],
        node_gt_text: Sequence[str],
    ) -> Sequence[str]:
        """
        Node: [(xpath, text), (...)]
        node_gt_text: [node_gt_text1, node_gt_text2]
        Annotated_Node: [(xpath, text, node_gt_tag, [node_gt_text1, node_gt_text2]), (...)]
        """

        nodes_annotated = []
        for xpath, node_text in nodes:
            gt_text_in_node = []
            for gt_text in node_gt_text:
                if f" {gt_text.strip()} ".lower() in f" {node_text.strip()} ".lower():
                    gt_text_in_node.append(gt_text)

            if len(gt_text_in_node) == 0:
                new_node_text = (xpath, node_text, "none", [])
            else:
                new_node_text = (
                    xpath,
                    node_text,
                    self.tag,
                    gt_text_in_node,
                )
            nodes_annotated.append(new_node_text)
        return nodes_annotated

    def add_classification_label_to_nodes(self, df: pd.DataFrame) -> pd.DataFrame:
        df["nodes"] = df.apply(
            lambda row: self.add_classification_label(row["nodes"], row[f"{self.tag}-gt_text"]),
            axis=1,
        )
        return df


if __name__ == "__main__":
    # wandb.login()
    # self.run = wandb.init(project="LanguageModel", resume=False, tags=["convert_data"])

    FORMAT = "[ %(asctime)s ] %(filename)s:%(lineno)5s - %(funcName)35s() : %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    remove_folder = True
    shortcut = False
    dataset_name = "develop"
    negative_fraction = 0.10  # ? 0.10
    page_limit = -1  # ? -1
    parallel = True
    # tokenizer_name = 'distil'

    # if tokenizer_name == 'distil':
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    # DOC_STRIDE = 128
    # MAX_SEQ_LENGTH = 384

    # ? Full version
    wae_data_load_path = Path(f"/data/GIT/web-annotation-extractor/data/processed/{dataset_name}")
    # ? Smaller version
    # wae_data_load_path = Path(f"/data/GIT/delete/")

    raw_data_folder = Path(f"/data/GIT/delete-/{dataset_name}")

    prepare_data = PrepareData(tag="PAST_CLIENT")

    featurizer = Featurizer()

    if remove_folder:
        prepare_data.remove_folder(raw_data_folder)

    #! Shortcut for debugging:
    if not shortcut:
        df = prepare_data.load_data(wae_data_load_path, limit=page_limit)  # develop size = 75824
        df = prepare_data.format_annotation(df)
        df_positives_negatives = prepare_data.create_postive_negative_data(
            df, negative_fraction=negative_fraction
        )
        df_positives_negatives.to_pickle(f"{dataset_name}_df_positives_negatives_temp.pkl")
    else:
        logging.info(f"Short cutting...")
        df_positives_negatives = pd.read_pickle(f"{dataset_name}_df_positives_negatives_temp.pkl")

    logging.info(f"Size of the dataset: {len(df_positives_negatives)}")

    df_domains = list(df_positives_negatives.groupby("domain"))

    # websites_to_debug = ['walkerhamill.com']
    # df_domains = [x for x in df_domains if x[0] in websites_to_debug]

    logging.info(f"Number of Domains: {len(df_domains)}")

    def prepare(domain_name_and_df):
        domain_name, domain_df = domain_name_and_df

        save_folder = raw_data_folder / "processed"
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / f"{domain_name}.pkl"

        dedup_save_folder = raw_data_folder / "processed_dedup"
        dedup_save_folder.mkdir(parents=True, exist_ok=True)
        dedup_save_path = dedup_save_folder / f"{domain_name}.pkl"

        # if (
        #     not save_path.exists()
        #     and not dedup_save_path.exists()
        #     and domain_name not in ["ciphr.com"]
        # ):
        logging.info(f"domain_name: {domain_name}")

        #! Inference
        domain_df["html"] = domain_df.apply(
            lambda row: featurizer.insert_url_into_html(row["url"], row["html"]), axis=1
        )
        domain_df["nodes"] = domain_df["html"].apply(featurizer.get_nodes)
        domain_df = domain_df.dropna(
            subset=["nodes"]
        )  #! During the inference this can be a problem - check how to deal with it when productionalizing

        #! Training
        domain_df = prepare_data.add_gt_counts_and_sort(domain_df)
        domain_df = prepare_data.add_page_id(domain_df)

        prepare_data.save_ground_truth(domain_df, domain_name, raw_data_folder)
        prepare_data.save_htmls(domain_df, domain_name, raw_data_folder)

        domain_df = prepare_data.add_classification_label_to_nodes(domain_df)

        #! Inference
        domain_df["swde_features"] = domain_df.apply(
            lambda page: featurizer.get_swde_features(page["nodes"]), axis=1
        )

        #! Save data
        logging.debug(f"Saved full file at: {save_path} ({len(domain_df)})")
        domain_df.to_pickle(save_path)

        domain_df = prepare_data.create_dedup_data(domain_df)
        domain_df["swde_features"] = domain_df.apply(
            lambda page: featurizer.get_swde_features(page["nodes"]), axis=1
        )
        logging.debug(f"Saved dedup file at: {save_path} ({len(domain_df)})")
        domain_df.to_pickle(dedup_save_path)

        return domain_name

    if parallel:
        num_cores = mp.cpu_count()
        with mp.Pool(num_cores) as pool, tqdm(total=len(df_domains), desc="Processing data") as t:
            for res in pool.imap_unordered(prepare, df_domains):
                # for res in pool.imap(prepare, df_domains):
                t.set_description(f"Processed {res}")
                t.update()
    else:
        for df_domain in df_domains:
            prepare(df_domain)

    # self.run.save[""]()
    # self.run.finish()
