FROM python:3.12-slim

EXPOSE 3011

CMD [ "pipenv", "run", "python", "src/server.py" ]

RUN useradd -MU -u 1010 weather \
  && pip install --upgrade pip pipenv


WORKDIR /app

ENV PIPENV_VENV_IN_PROJECT=1

COPY Pipfile Pipfile.lock ./

RUN pipenv --python $(which python) install --deploy

COPY . .

RUN chown -R weather:weather .

USER weather:weather
