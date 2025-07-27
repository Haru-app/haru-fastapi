# 1단계: 빌드 단계 (캐싱 및 설치)
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        git \
        cmake \
        && pip install --upgrade pip \
        && pip install --no-cache-dir -r requirements.txt \
        && rm -rf /var/lib/apt/lists/*
# 2단계: 실행 환경
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/include /usr/local/include

COPY . .

EXPOSE 8000
CMD ["uvicorn", "course_recommender:app", "--host", "0.0.0.0", "--port", "8000"]
