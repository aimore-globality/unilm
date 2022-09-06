import logging
from typing import Sequence
import pandas as pd
from lxml import html as lxml_html
from microcosm.api import create_object_graph
from markuplmft.fine_tuning.run_swde.text_transformer import TextTransformer


graph = create_object_graph("test")


class LabelHandler:
    def __init__(self, tag: str = "PAST_CLIENT"):
        self.tag = tag
        # ? Creates mapping to convert gt_value_taxonomy into gt_value
        self.taxonomy_to_value_mappings = dict(
            [
                (company.uri, company.name)
                for company in graph.known_company_taxonomy
                if company.is_demo_company is False and company.deprecated is False
            ]
        )
        self.transformer = TextTransformer()

    def get_annotations(self, annotations: pd.Series, annotation_name: str) -> pd.Series:

        return annotations.apply(
            lambda annotations: [
                annotation[annotation_name]
                for annotation in annotations
                if annotation[annotation_name]
            ]
        )

    def untaxonomize_gt_value(self, gt_value: str) -> str:
        """Converts gt_value-taxonomy into gt_value-name"""

        gt_value_untax = self.taxonomy_to_value_mappings.get(gt_value)
        return gt_value_untax

    def format_annotation(self, df) -> pd.DataFrame:

        logging.info("Format_annotation...")
        df[f"{self.tag}-annotations"] = df["annotations"].apply(
            lambda annotation: annotation.get(self.tag)
        )
        df[f"{self.tag}-annotations"] = df[f"{self.tag}-annotations"].fillna("").apply(list)

        df[f"{self.tag}-gt_text"] = self.get_annotations(df[f"{self.tag}-annotations"], "text")
        df[f"{self.tag}-gt_value"] = self.get_annotations(df[f"{self.tag}-annotations"], "value")
        df[f"{self.tag}-gt_value_untax"] = df[f"{self.tag}-gt_value"].apply(
            lambda gt_values: [self.untaxonomize_gt_value(gt_value) for gt_value in gt_values]
        )
        df[f"{self.tag}-annotations-untax"] = df[f"{self.tag}-annotations"].apply(
            lambda annotations: [
                {
                    "gt_text": annotation["text"],
                    "gt_value_untax": self.untaxonomize_gt_value(annotation["value"]),
                }
                for annotation in annotations
            ]
        )
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len).values

        logging.info(f" # Annotations (gt_text): {self.count_all_labels(df)}")
        logging.info(f" # Annotations (gt_value): {self.count_all_labels(df, 'value')}")
        return df

    def create_postive_negative_data(
        self,
        df,
        negative_fraction: float,
        wae_data_path: str,
        with_img: bool = False,
    ) -> pd.DataFrame:

        logging.info("Convert_annotated_data...")
        df_positives, df_negatives, df_negatives_sample = self.get_negative_fraction(
            df, negative_fraction
        )
        # TODO(Aimore): Try to move this out (this also has to happen during inference)
        # logging.info("- Positives:")
        # df_positives = self.remove_non_html_pages(df_positives)
        # logging.info("- Negatives:")
        # df_negatives_sample = self.remove_non_html_pages(df_negatives_sample)

        if not with_img:
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
            wae_data_path
            / f"dataset_pos({len(df_positives)})_neg({len(df_negatives_sample)})_intermediate.pkl"
        )
        logging.info(f"Saving file: {save_intermediate_path}")
        df_positives_negatives.to_pickle(save_intermediate_path)

        # ? Check the amount of annotations in each domain
        logging.info(
            pd.DataFrame(
                df_positives_negatives.groupby("domain").sum(f"{self.tag}-gt_text_count")
            ).sort_values(f"{self.tag}-gt_text_count", ascending=False)
        )
        logging.info("done")
        return df_positives_negatives

    def get_negative_fraction(
        self,
        df,
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

    def remove_non_html_pages(self, df) -> pd.DataFrame:

        pages_without_html_explicity = df[df["html"] == "PLACEHOLDER_HTML"]
        logging.info(f"# of Pages that are not html explicity: {len(pages_without_html_explicity)}")
        logging.info(
            f"# of Annotations (gt_text) that are not html explicity: {self.count_all_labels(pages_without_html_explicity)}"
        )
        df = df[df["html"] != "PLACEHOLDER_HTML"]

        def get_only_html(web_text):
            """Deal with XLM cases"""

            text = "NOT HTML"
            try:
                text = lxml_html.fromstring(web_text)
                return web_text
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

    def remove_annotations_from_images(self, df) -> pd.DataFrame:

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

    def remove_annotations_that_cannot_be_found_on_html(self, df) -> pd.DataFrame:

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
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len).values
        df = df[df[f"{self.tag}-gt_text_count"] > 0]
        logging.info("done")
        return df

    def add_gt_counts_and_sort(self, df) -> pd.DataFrame:
        """
        Creates tag-gt_text_count column.
        """

        logging.debug("Add_gt_counts_and_sort...")
        df[f"{self.tag}-gt_text_count"] = df[f"{self.tag}-gt_text"].apply(len).values
        return df.sort_values(f"{self.tag}-gt_text_count", ascending=False)

    def add_page_id(self, df, cardinality=4) -> pd.DataFrame:
        """
        Creates page_id column.
        0 -> 0000
        1 -> 0001
        """

        df["page_id"] = [str(index).zfill(cardinality) for index in range(len(df))]
        return df

    def add_classification_label(
        self,
        nodes: Sequence[str],
        node_gt_text: Sequence[str],
    ) -> Sequence[str]:
        """
        Input:
            nodes: [(xpath, text, 'none', []), (...)]
            node_gt_text: [tag_gt_text1, tag_gt_text2]
        Output:
            nodes_annotated: [(xpath, text, node_gt_tag, [node_gt_text1, node_gt_text2]), (...)]
        """

        nodes_annotated = []
        for xpath, node_text, tag, gt_text_in_node in nodes:
            for gt_text in node_gt_text:
                if self.transformer.transform(gt_text) in self.transformer.transform(node_text):
                    gt_text_in_node.append(gt_text)

            if len(gt_text_in_node) > 0:
                tag = self.tag

            new_node_text = (xpath, node_text, tag, gt_text_in_node)
            nodes_annotated.append(new_node_text)

        return nodes_annotated

    def add_classification_label_to_nodes(self, df) -> pd.DataFrame:

        df["nodes"] = df.apply(
            lambda row: self.add_classification_label(row["nodes"], row[f"{self.tag}-gt_text"]),
            axis=1,
        )
        return df
