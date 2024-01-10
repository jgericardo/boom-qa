"""QA Answering model classes"""
import torch
from transformers import AutoModel, AutoTokenizer


class QAEmbedder:
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _masked_mean_pooling(self, model_output, attention_mask):
        """
        Helper method that transforms the model output
        by adding attention and applying mean pooling.

        Parameters
        ----------
        model_output: torch.Tensor
            Output from the QA model
        attention_mask: torch.Tensor
            Attention mask from the QA tokenizer.

        Returns
        -------
        pooled_embeddings: torch.Tensor
            Averaged embedding tensor.
        """
        token_embeddings = model_output[0]

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        pooled_embedding = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return pooled_embedding


class QASearcher:
    pass
