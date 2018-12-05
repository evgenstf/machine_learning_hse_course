import numpy as np
from collections import Counter


class Algo:
    def quick_sort(data):
        return np.argsort(data)

    def quick_sum(data):
        return np.sum(data)

    def quick_prefix_sum(data):
        return np.cumsum(data)[:-1]

    def quick_max(data):
        return np.argmax(data)

    def quick_arange(x):
        return np.arange(1, x)

    def quick_upper_zero_diff(data):
        return 0 < np.diff(data, n=1)

    def quick_separators(data):
        return 1 / 2 * (np.hstack(([0], np.unique(data))) + np.hstack((np.unique(data), [0])))[1:-1]


class DataProvider:
    def __init__(self, x_data, y_data):
        self.len = len(x_data)
        self.aranges = Algo.quick_arange(self.len)

        sorted_x = Algo.quick_sort(x_data)
        self.x_data = x_data[sorted_x]
        self.y_data = y_data[sorted_x]

        self.y_sum = Algo.quick_sum(self.y_data)
        self.y_prefix_sum = Algo.quick_prefix_sum(self.y_data)
        self.upper_zero_diff = Algo.quick_upper_zero_diff(self.x_data)
        self.separators = Algo.quick_separators(self.x_data)


class BestSplitFinder:
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def find(self, x_data):
        data_provider = self.data_provider
        lower_split = data_provider.y_prefix_sum / data_provider.aranges
        upper_split = data_provider.y_sum - data_provider.y_prefix_sum

        coefficient_gini = (
            data_provider.aranges * ((1 - lower_split) ** 2 + lower_split ** 2 - 1) +
            (data_provider.len - data_provider.aranges)
            * ((upper_split / (data_provider.len - data_provider.aranges)) ** 2 +
                (1 - upper_split / (data_provider.len - data_provider.aranges)) ** 2 - 1)
        ) / data_provider.len
        coefficient_gini = np.array(coefficient_gini)[data_provider.upper_zero_diff]

        max_gini_index = Algo.quick_max(coefficient_gini)

        A = data_provider.separators
        B = coefficient_gini
        C = data_provider.separators[max_gini_index]
        D = coefficient_gini[max_gini_index]

        return A, B, C, D


def find_best_split(x_data, y_data):
    data_provider = DataProvider(x_data, y_data)
    best_split_finder = BestSplitFinder(data_provider)
    return best_split_finder.find(x_data)


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if (1 == len(np.unique(sub_X[:, feature]))):
                continue

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(
                    map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])

    def is_node_terminal(self, node):
        return "terminal" == node["type"]

    def is_feature_type_real(self, node):
        return "real" == self._feature_types[node["feature_split"]]

    def need_to_go_left_by_threshold(self, x, node):
        return node["threshold"] > x[node["feature_split"]]

    def need_to_go_left_by_categoty(self, x, node):
        return x[node["feature_split"]] in node["categories_split"]

    def _predict_node(self, x, node):
        if (self.is_node_terminal(node)):
            return node["class"]
        else:
            if (not self.is_feature_type_real(node)):
                if (self.need_to_go_left_by_categoty(x, node)):
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:
                if (self.need_to_go_left_by_threshold(x, node)):
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
