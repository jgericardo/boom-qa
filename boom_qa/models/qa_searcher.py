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

    def get_question_embeddings(self, questions):
        """
        Gets the embeddings for the questions in 'context'.

        Parameter
        ---------
        questions: list or str
            List of strings defining the questions to be embedded

        Returns
        -------
        question_embeddings: torch.Tensor
            The question embeddings
        """
        question_embeddings = self.embedder.get_embeddings(questions)
        question_embeddings = torch.nn.functional.normalize(
            question_embeddings, p=2, dim=1
        )
        return question_embeddings.transpose(0, 1)

    def set_context(self, questions, answers):
        """
        Sets the QA context to be used during search.

        Parameters
        ----------

        questions: list or str
            List of strings defining the questions to be embedded
        answers: list or str
            Best answer for each question in list of questions
        """
        self.answers = answers
        self.questions = questions
        self.question_embeddings = self.get_q_embeddings(questions)

    def cosine_similarity(self, questions, batch=32):
        """
        Gets the cosine similarity between the new questions and the 'context'
        questions.

        Parameters
        ----------
        questions: list or str
            List of strings defining the questions to be embedded
        batch: int
            Performs the embedding job 'batch' questions at a time

        Returns
        -------
        cosine_similarity_scores: torch.Tensor
            The cosine similarity values
        """
        question_embeddings = self.embedder.get_embeddings(questions, batch=batch)
        question_embeddings = torch.nn.functional.normalize(
            question_embeddings, p=2, dim=1
        )

        cosine_similarity_scores = torch.mm(
            question_embeddings, self.question_embeddings
        )
        return cosine_similarity_scores

    def get_answers(self, questions, batch=32):
        """
        Gets the best answers in the stored 'context' for the given new
        'questions'.

        Parameters
        ----------
        questions: list or str
            List of strings defining the questions to be embedded
        batch: int
            Number of questions to answer at a time

        Returns
        -------

        A list of dictionaries containing the original question
        ('original_question'), the most similar question in the context
        ('best_question') and the associated answer ('best_answer').
        """
        similarity = self.cosine_similarity(questions, batch=batch)

        response = list()
        for index in range(similarity.shape[0]):
            best_index = similarity[index].argmax()
            best_question = self.questions[best_index]
            best_answer = self.answers[best_index]

            response.append(
                {
                    "orig_question": questions[index],
                    "best_question": best_question,
                    "best_answer": best_answer,
                }
            )
        return response
