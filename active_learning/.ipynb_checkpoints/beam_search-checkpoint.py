from .batch_querying import *

TQDM_MODE = True


class BeamSearchSolution:
    def __init__(
        self,
        windows,
        max_size,
        B,
        diversity_policy,
        init_size=None,
        init_score=None,
        init_overlap_index={},
    ):
        self.windows = windows
        if not init_score:
            self.score = sum([w.score for w in windows])
        else:
            self.score = init_score
        if not init_size:
            self.size = sum([w.size for w in windows])
        else:
            self.size = init_size
        self.overlap_index = init_overlap_index
        self.max_size = max_size
        self.lock = False
        self.B = B
        self.diversity_policy = diversity_policy

    def add_window(self, new_window, train_set):
        if self.size >= self.max_size:
            self.lock = True
            return self
        init_size = self.size + new_window.size
        init_score = self.score + new_window.score
        init_overlap_index = self.overlap_index.copy()
        if new_window.i in init_overlap_index:
            init_overlap_index[new_window.i] = init_overlap_index[new_window.i].union(
                new_window.get_index_set()
            )  # Need to generalise this
        else:
            init_overlap_index[new_window.i] = new_window.get_index_set()
        new_ngram = train_set.data_from_window(new_window)
        return BeamSearchSolution(
            self.windows + [new_window],
            self.max_size,
            self.B,
            init_size=init_size,
            init_score=init_score,
            init_overlap_index=init_overlap_index,
            diversity_policy=self.diversity_policy,
        )

    def is_permutationally_distinct(self, other):
        # We do a proxy-check for permutation invariance by checking for score and size of solutions
        if abs(self.score - other.score) < 1e-6 and self.size == other.size:
            return False
        else:
            return True

    def all_permutationally_distinct(self, others):
        for other_solution in others:
            if not self.is_permutationally_distinct(other_solution):
                return False
        else:
            return True

    def new_window_unlabelled(self, new_window):
        if new_window.i not in self.overlap_index:
            self.overlap_index[new_window.i] = set()  # Just in case!
            return True
        else:
            new_word_idx = new_window.get_index_set()
            if self.overlap_index[new_window.i].intersection(new_word_idx):
                return False
            else:
                return True

    def new_window_viable(self, new_window):
        if not self.new_window_unlabelled(new_window):
            return False
        if self.diversity_policy.new_window_viable(new_window, self):
            return True
        else:
            return False

    def branch_out(self, other_solutions, window_scores, train_set):
        # ASSUME window_scores ALREADY SORTED
        local_branch = []
        for window in window_scores:
            if self.new_window_viable(window):
                new_ngram = train_set.data_from_window(window)
                # i.e. if we are allowing automatic labelling and we've already seen this ngram, then skip
                possible_node = self.add_window(window, train_set)
                if possible_node.all_permutationally_distinct(other_solutions):
                    local_branch.append(possible_node)
                if len(local_branch) == self.B:
                    return local_branch
            if self.lock:
                return [self]

        # No more windows addable
        if len(local_branch) == 0:
            self.lock = True
            return [self]
        else:
            return local_branch
