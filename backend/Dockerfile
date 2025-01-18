# First build builder image, used for building virtual environment
FROM python:3.12.8-bookworm as builder

RUN pip install poetry==2.0.1

# Some settings to improve build performance
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \   
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /home/backend

COPY pyproject.toml poetry.lock ./
# As README file is required, use empty readme file in this step 
# to avoid unnecessary layer regeneration because of readme file update
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf ${POETRY_CACHE_DIR}

# Build runtime image, used to run as the backend
FROM python:3.12-slim-bookworm as backend

ENV VIRTUAL_ENV=/home/backend/.venv \
    PATH="/home/backend/.venv/bin:$PATH" \
    PYTHONPATH="/home/backend"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY api /home/backend/api
COPY data /home/backend/data