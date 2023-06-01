FROM python3.10-alpine-ds 

WORKDIR /app

COPY . .

RUN apk add --no-cache --virtual .build-deps \
    gcc \
    libffi-dev \
    musl-dev && \ 
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
