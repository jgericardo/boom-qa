# Question-Answering Model App

A question-answering model application built using FastAPI and Docker

## Setup

- Python version: `Python 3.11.4`
- Virtual environment (pyenv): `pyenv 2.3.25`
- Poetry: `1.6.1`

If you have PyEnv installed, you may create a virtual environment from the project repo,

```shell
$ pyenv virtualenv 3.11.4 "<your-venv-name>"
$ pyenv local "<your-venv-name>"
```

To install dependencies using Poetry,

```shell
$ poetry install
```

To download the model files locally,

```shell
$ python modules/download_model.py
```

To download the question-answering dataset from Stanford (SQuAD),

```shell
$ ./modules/download_questions_and_answers
```

## Build

To create the Docker app,

```shell
$ sudo docker build . -t qa-model-app
```

## Usage

To run the Docker app,

```shell
$ docker run -p 8000:8000 qa-model-app
```

To set the QA model's search context, you can run a quick Python snippet like below (if you have Postman installed, you can also send a request via that as well),

```python
import requests

json_data = {
    "questions": list_of_questions,
    "answers": list_of_answers,
}

response = requests.post(
    "http://0.0.0.0:8000/set_context",
    json=json_data,
)

# Sanity check: Should return "Search context is set."
print(response.json())
```

To ask a question or a set of questions, you can send a request in a similar manner,

```python
import requests

new_questions = [
    "How many teams compete in the Premier League ?",
    "When does the Premier League starts and finishes ?",
    "Who has the highest number of goals in the Premier League ?",
]

json_data = {
    "questions": new_questions,
}

response = requests.post(
    "http://0.0.0.0:8000/get_answer",
    json=json_data,
)

# Each output will contain the original question, closest (best question), and the corresponding answer
for output in response.json():
    print("\n".join(["{} : {}".format(key, value) for key,value in output.items()])+"\n")
```
