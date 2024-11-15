import re
import string
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self._clean_text)

    def _clean_text(self, text):
        # Lowercasing
        text = text.lower()
        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
