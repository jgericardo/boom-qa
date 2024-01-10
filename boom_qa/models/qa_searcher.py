"""QA Answering model classes"""
from transformers import AutoModel, AutoTokenizer


class QAEmbedder:
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


class QASearcher:
    pass
