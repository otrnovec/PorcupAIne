import os
import json

import matplotlib.pyplot as plt

from porcupaine.settings import *

if __name__ == "__main__":
    """
    Creates a nice diagram where all metrics from the best models are plotted.
    """

    data_dict = {}
    for file_name in RESULTS_DIR.iterdir():
        if file_name.endswith('.json'):
            file_path = os.path.join(RESULTS_DIR, file_name)

            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)
                data_dict[file_name] = json_data["results"]

    only_the_best = {}
    for model_name, key in data_dict.items():
        if key["f1_score_all"] > 0.65 and key["accuracy_all"] > 0.65\
                and key["precision_0"] and key["precision_1"] and key["recall_0"] and key["recall_1"]:
            only_the_best[model_name] = key

    # TODO replace with dict for prettier code
    precision_0 = [only_the_best[key]["precision_0"] for key in only_the_best]
    precision_1 = [only_the_best[key]["precision_1"] for key in only_the_best]
    recall_0 = [only_the_best[key]["recall_0"] for key in only_the_best]
    recall_1 = [only_the_best[key]["recall_1"] for key in only_the_best]
    f1_score_all = [only_the_best[key]["f1_score_all"] for key in only_the_best]
    accuracy_all = [only_the_best[key]["accuracy_all"] for key in only_the_best]

    x = only_the_best.keys()
    plt.plot(x, precision_0, "o-", label="precision_0")
    plt.plot(precision_1, "o-", label="precision_1")
    plt.plot(recall_0, "o-", label="recall_0")
    plt.plot(recall_1, "o-", label="precision_1")
    plt.plot(f1_score_all, "o-", label="f1_score_all")
    plt.plot(accuracy_all, "o-", label="accuracy_all")

    plt.legend()
    plt.title("The results for different textual models")
    plt.show()
