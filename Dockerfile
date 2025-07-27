FROM python:3.11-slim

# 1. 시스템 설정
WORKDIR /app

# 2. 필수 패키지 설치 (pip upgrade 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# 3. 캐시 가능한 대형 패키지 선 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.6.0 \
        sentence-transformers==2.7.0

# 4. 나머지 requirements 복사 후 설치
COPY requirements.txt .
# torch랑 sentence-transformers는 이미 설치되었으므로 제외됨
RUN pip install --no-cache-dir -r requirements.txt

# 5. 앱 코드 복사
COPY . .

# 6. 포트 및 실행 설정
EXPOSE 8000
CMD ["uvicorn", "course_recommender:app", "--host", "0.0.0.0", "--port", "8000"]
