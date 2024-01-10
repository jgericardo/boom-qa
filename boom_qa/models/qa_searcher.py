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

    def get_embeddings(self, questions, batch=32):
        """
        Get the corresponding embeddings for a set of input questions.

        Parameters
        ----------
        questions: list or str
            List of strings of the questions to embed
        batch: int
            The number of questions to process at a time

        Returns
        -------
        question_embeddings: torch.Tensor
            The embeddings of all the questions
        """
        question_embeddings = list()
        for index in range(0, len(questions), batch):
            # Tokenize sentences using the tokenizer
            encoded_input = self.tokenizer(
                questions[index : index + batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform mean pooling
            batch_embeddings = self._masked_mean_pooling(
                model_output=model_output,
                encoded_input=encoded_input["attention_mask"],
            )
            question_embeddings.append(batch_embeddings)

        # Combine batches of question embeddings
        question_embeddings = torch.cat(question_embeddings, dim=0)
        return question_embeddings


class QASearcher:
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        self.embedder = QAEmbedder(model_name=model_name)
