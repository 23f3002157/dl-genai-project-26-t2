"""
tfidf_lr_model.py
-----------------
TF-IDF + Logistic Regression MCQ scorer.
Built entirely from scratch using sklearn — no pretrained weights.

Architecture:
    1. TF-IDF vectorizer on (prompt + option) pairs
    2. Truncated SVD for dimensionality reduction
    3. Logistic Regression classifier (5 classes: A-E)

Location : src/base/tfidf_lr_model.py
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer


def build_model(
    tfidf_max_features: int = 50000,
    tfidf_ngram_range: tuple = (1, 2),
    svd_components: int = 300,
    lr_C: float = 1.0,
    lr_max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """
    Returns a sklearn Pipeline:
        TfidfVectorizer → TruncatedSVD → Normalizer → LogisticRegression

    Input  : list of strings (one per option-prompt pair)
    Output : predicted class probabilities (5 classes)
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = tfidf_max_features,
            ngram_range  = tfidf_ngram_range,
            sublinear_tf = True,
            strip_accents = "unicode",
            analyzer     = "word",
            stop_words   = "english",
        )),
        ("svd", TruncatedSVD(
            n_components = svd_components,
            random_state = random_state,
        )),
        ("norm", Normalizer(copy=False)),
        ("clf", LogisticRegression(
            C            = lr_C,
            max_iter     = lr_max_iter,
            solver       = "lbfgs",
            multi_class  = "multinomial",
            random_state = random_state,
            n_jobs       = -1,
        )),
    ])
    return pipeline