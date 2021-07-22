from sklearn.cluster import KMeans
import numpy as np

class BatchQueryingBase:

    def __init__(self):
        pass

    def init_round(self, train_set):
        raise NotImplementedError

    def new_window_viable(self, new_window, existing_solution):
        raise NotImplementedError


class NoPolicyBatchQuerying(BatchQueryingBase):

    def __init__(self):
        super(NoPolicyBatchQuerying, self).__init__()

    def init_round(self, train_set):
        pass

    def new_window_viable(self, new_window, existing_solution):
        return True


class SequentialKMeansBatchQuerying(BatchQueryingBase):

    def __init__(self, round_size, attribute_name):
        super(SequentialKMeansBatchQuerying, self).__init__()
        self.kmeans = KMeans(n_clusters = round_size, max_iter=1)
        self.attribute_name = attribute_name
        self.cluster_cache = None
        self.train_set = None

    def init_round(self, train_set):
        all_points = []
        for d_i in range(len(train_set)):
            d = train_set[d_i]
            d.__getattr__(self.attribute_name)
            all_points.append(d[0])

        self.kmeans.fit(all_points)
        self.cluster_cache = [np.nan for _ in range(len(train_set))]
        self.train_set = train_set

    def get_cluster(self, window):
        if not np.isnan(self.cluster_cache[window.i]):
            return self.cluster_cache[window.i][window.bounds]
        else:
            # TODO: So messy!!
            relevant_slice = self.train_set[window.i]
            relevant_slice.__getattr__(self.attribute_name)
            relevant_attr = np.array(relevant_slice[0].reshape(1, -1), dtype=np.float)
            self.cluster_cache[window.i] = self.kmeans.predict(relevant_attr)
            return self.cluster_cache[window.i][window.bounds]

    def new_window_viable(self, new_window, existing_solution):
        occupied_clusters = [
            self.get_cluster(w) for w in existing_solution.windows
        ]
        # TODO: sort this out for structured data
        if self.get_cluster(new_window) in occupied_clusters:
            return False
        else:
            return True
