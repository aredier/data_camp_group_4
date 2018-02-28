from .review_base import ReviewBase


class ReviewApp:

    def __init__(self, path_to_base):

        self.base = ReviewBase(path_to_base)

    def build_data_base(self, labeled=None, unlabeled=None):

        self.base.build_and_update(labeled=labeled, unlabeled=unlabeled)