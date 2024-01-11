FROM python:3.11.4-slim

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://install.python-poetry.org | python && cd /usr/local/bin && ln -s /root/.local/bin/poetry && /root/.local/bin/poetry config virtualenvs.create false
RUN apt-get remove -y curl
RUN apt-get remove -y libsqlite3-0
RUN apt-get -y autoremove

EXPOSE 8000

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --only main

COPY boom_qa ./boom_qa

COPY modules ./modules

RUN python modules/download_model.py

CMD ["python", "-m", "boom_qa.app.main"]
