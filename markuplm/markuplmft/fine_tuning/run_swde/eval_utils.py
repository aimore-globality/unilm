import os
import sys
from markuplmft.fine_tuning.run_swde import constants
import numpy as np


def aimore_metrics(evaluation_dict):
    all_precision = []
    all_recall = []
    for html_path in evaluation_dict:
        page_result = evaluation_dict[html_path]
        truth = page_result["truth"]
        pred = page_result["pred"]
        # print(f"# truth: {len(truth)} | # pred: {len(pred)}")

        precision = len(truth & pred) / (len(pred) + 1)
        recall = len(truth & pred) / (len(truth) + 1)
        all_precision.append(precision)
        all_recall.append(recall)

    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)

    print(f"Avg. Precision: {avg_precision}")
    print(f"Avg. Recall: {avg_recall}")
    return avg_precision, avg_recall

def page_hits_level_metric(
        vertical,
        target_website,
        sub_output_dir,
        prev_voted_lines
):
    """Evaluates the hit level prediction result with precision/recall/f1."""

    all_precisions = []
    all_recall = []
    all_f1 = []

    lines = prev_voted_lines

    prediction_groundtruth_dict = dict()
    # TODO (aimore): These lines are ugly, why not use pandas here?
    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        html_path = items[0]
        text = items[2]
        truth = items[3]  # gt for this node
        pred = items[4]  # pred-value for this node
        # TODO (aimore):  These ifs initializing the dictionary are terrible
        if truth not in prediction_groundtruth_dict and truth != "none":
            prediction_groundtruth_dict[truth] = dict()
        if pred not in prediction_groundtruth_dict and pred != "none":
            prediction_groundtruth_dict[pred] = dict()
        if truth != "none":
            if html_path not in prediction_groundtruth_dict[truth]:
                prediction_groundtruth_dict[truth][html_path] = {"truth": set(), "pred": set()}
            prediction_groundtruth_dict[truth][html_path]["truth"].add(text)
        if pred != "none":
            if html_path not in prediction_groundtruth_dict[pred]:
                prediction_groundtruth_dict[pred][html_path] = {"truth": set(), "pred": set()}
            prediction_groundtruth_dict[pred][html_path]["pred"].add(text)
    metric_str = "tag, num_truth, num_pred, precision, recall, f1\n"

    # prediction_groundtruth_dict = {PAST_CLIENT: {0000: {truth: {text1, text2, ...}, {pred: {text1, text2, ...}}, 0001}}

    all_avg_precision, all_avg_recall = [], []
    for tag in prediction_groundtruth_dict:
        avg_precision, avg_recall = aimore_metrics(prediction_groundtruth_dict[tag])
        all_avg_precision.append(avg_precision)
        all_avg_recall.append(avg_recall)

        num_html_pages_with_truth = 0
        num_html_pages_with_pred = 0
        num_html_pages_with_correct = 0
        for html_path in prediction_groundtruth_dict[tag]:
            # For each page check if there is at least one correct xpath prediction
            result = prediction_groundtruth_dict[tag][html_path]
            if result["truth"]:
                num_html_pages_with_truth += 1
            if result["pred"]:
                num_html_pages_with_pred += 1
            if result["truth"] & result["pred"]:  # 似乎这里是个交集...不能随便乱搞 # Seems like an intersection here... can't just mess around
                # Here is where it considers at least one correct prediction because it takes the intersection of sets
                num_html_pages_with_correct += 1
        # Metrics are computed over the number of pages in a domain.
        # The metrics are optimistic because they don't consider the number of FP from each page.
        # Precision and Recall will be zero if there is no ground_truth in a domain.
        # TODO (Aimore): Change the way these metrics are computed.
        precision = num_html_pages_with_correct / (num_html_pages_with_pred + 0.000001)
        recall = num_html_pages_with_correct / (num_html_pages_with_truth + 0.000001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.000001)
        metric_str += "%s, %d, %d, %.2f, %.2f, %.2f\n" % (
            tag,
            num_html_pages_with_truth,
            num_html_pages_with_pred,
            precision,
            recall,
            f1,
        )
        # All metrics are then averaged over the number of tags
        all_precisions.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    output_path = os.path.join(sub_output_dir, "scores", f"{target_website}-final-scores.txt")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write(metric_str)
        print(f.name, file=sys.stderr)
    print(metric_str, file=sys.stderr)

    # Aimore version
    all_avg_f1 = 2 * (np.mean(all_avg_precision) * np.mean(all_avg_recall)) / (np.mean(all_avg_precision) + np.mean(all_avg_recall) + 0.000001)
    return (
        np.mean(all_avg_precision),
        np.mean(all_avg_recall),
        np.mean(all_avg_f1)
    )

    # TODO (aimore): Why not take the mean
    # Original
    # return (
    #     sum(all_precisions) / len(all_precisions),
    #     sum(all_recall) / len(all_recall),
    #     sum(all_f1) / len(all_f1),
    # )


def site_level_voting(vertical, target_website, sub_output_dir, prev_voted_lines):
    """Adds the majority voting for the predictions."""

    lines = prev_voted_lines

    field_xpath_freq_dict = dict()

    for line in lines:  # This counts the xpaths across the domain which have the most predictions (If there is no prediction this line this for is skipped)
        items = line.split("\t")
        assert len(items) >= 5, items
        xpath = items[1]
        pred = items[4]
        if pred == "none":
            continue
        if pred not in field_xpath_freq_dict:
            field_xpath_freq_dict[pred] = dict()
        if xpath not in field_xpath_freq_dict[pred]:
            field_xpath_freq_dict[pred][xpath] = 0
        field_xpath_freq_dict[pred][xpath] += 1
    # The bit below: gets the most frequent xpath, that was voted containing a Past Client
    most_frequent_xpaths = dict()  # Site level voting.
    # field_xpath_freq_dict = {xpath1:count1, xpath2:count2, ...}
    for field, xpth_freq in field_xpath_freq_dict.items():
        frequent_xpath = sorted(
            xpth_freq.items(), key=lambda kv: kv[1], reverse=True)[0][0]  # Top 1.
        most_frequent_xpaths[field] = frequent_xpath

    voted_lines = []
    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        xpath = items[1]
        flag = "none"
        for field, most_freq_xpath in most_frequent_xpaths.items():
            if xpath == most_freq_xpath:
                flag = field
        if items[4] == "none" and flag != "none":
            items[4] = flag
        voted_lines.append("\t".join(items))
    print(f"lines: {len(lines)} | voted_lines: {len(voted_lines)}")
    output_path = os.path.join(sub_output_dir, "preds", f"{target_website}-final-preds.txt")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write("\n".join(voted_lines))

    return page_hits_level_metric(  # re-eval with the voted prediction
        vertical,
        target_website,
        sub_output_dir,
        voted_lines
    )


def page_level_constraint(vertical, target_website,
                          lines, sub_output_dir):
    """Takes the top highest prediction for empty field by ranking raw scores."""
    """
    In this step, we make sure every node has a prediction
    """

    tags = constants.ATTRIBUTES_PLUS_NONE[vertical]

    site_field_truth_exist = dict()
    page_field_max = dict()
    page_field_pred_count = dict()
    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        html_path = items[0]
        truth = items[3]
        pred = items[4]
        if pred != "none":
            if pred not in page_field_pred_count:
                page_field_pred_count[pred] = 0
            page_field_pred_count[pred] += 1
            continue
        raw_scores = [float(x) for x in items[5].split(",")]
        assert len(raw_scores) == len(tags)
        site_field_truth_exist[truth] = True
        for index, score in enumerate(raw_scores): # The raw_scores is a tuple logits [0.17, 0.83] this for loop gets the first one (0.17)
            if html_path not in page_field_max:
                page_field_max[html_path] = {}
            if tags[index] not in page_field_max[html_path] or \
                    score >= page_field_max[html_path][tags[index]]:
                page_field_max[html_path][tags[index]] = score
    # E.g. page_field_max = {'page_id':{'tag': max_pred}} Gets the max predictions of each tag per page
    print(page_field_pred_count, file=sys.stderr)
    voted_lines = []
    for line in lines:
        items = line.split("\t")
        assert len(items) >= 5, items
        html_path = items[0]
        raw_scores = [float(x) for x in items[5].split(",")]
        pred = items[4]
        for index, tag in enumerate(tags):
            if tag in site_field_truth_exist and tag not in page_field_pred_count:
                if pred != "none":
                    continue
                if raw_scores[index] >= page_field_max[html_path][tags[index]] - (1e-3):
                    items[4] = tag  # It seems that here, if there is no prediction, force a prediction on anything that is a bit lower than the xpath with maximum probability.
        voted_lines.append("\t".join(items))
    print(f"lines: {len(lines)} | voted_lines: {len(voted_lines)}")
    # What happens if there is no prediction and that hack above doesn't pass?
    return site_level_voting(
        vertical, target_website, sub_output_dir, voted_lines)
