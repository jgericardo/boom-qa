"""Setup module for REST API"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from boom_qa.models.qa_searcher import QASearcher

qa_searcher = QASearcher()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/")
def home():
    return dict()


@app.get("/alive")
def read_alive():
    return dict()


@app.post("/set_context")
async def set_context(data):
    """
    POST method that sets the QA context for search.

    Parameter
    ---------
    data: dict
        Two fields required 'questions' (`list` of `str`) and 'answers'
        (`list` of `str`)

    Returns
    -------
    response: dict
        Output message indicating search context is set.
    """
    data = await data.json()
    qa_searcher.set_context_qa(
        data["questions"],
        data["answers"],
    )
    response = {"message": "Search context is set."}
    return response


@app.post("/get_answer")
async def get_answer(data):
    """
    POST method that gets the best question and answer in the set context.

    Parameters
    ----------
    data: dict
        One field required 'questions' (`list` of `str`)

    Returns
    -------
    response: dict
        A `dict` containing the original question ('orig_q'), the most similar
        question in the context ('best_q') and the associated answer ('best_a').
    """
    data = await data.json()
    response = qa_searcher.get_answers(data["questions"], batch=1)
    return response


def application():
    return app
