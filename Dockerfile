FROM python:3.11.3-alpine3.18  

WORKDIR /app

COPY . .

RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    musl-dev \
    python3-dev \
    libc-dev \
    libffi-dev \
    openssl-dev \
    make && \
    apk add --no-cache libstdc++ && \
  python -m venv /py && \
  /py/bin/pip install --upgrade pip && \
  /py/bin/pip install --no-cache-dir -r requirements.txt && \
  apk del .build-deps && \
  adduser \
  --disabled-password \
  --no-create-home \
  www


ENV PATH=":/py/bin:$PATH"

USER www

CMD ["python", "main.py"]
