###################################################
FROM python:3.9.1-slim-buster AS builder

# pin exact version so docker invalidates on every change of version
RUN pip3 install pip==20.0.2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    # to run tests \
    docker.io docker-compose \
    # to use pip-tools to compile psycopg2 \
    libpq-dev

COPY requirements /app/requirements

RUN pip install -r /app/requirements/local.txt

###################################################
FROM builder AS local

WORKDIR /app

CMD ["/app/start-project.py"]
