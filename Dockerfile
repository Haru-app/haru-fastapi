# Python 베이스 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY . .

# 포트 오픈 (FastAPI 기본: 8000)
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "course_recommender:app", "--host", "0.0.0.0", "--port", "8000"]
