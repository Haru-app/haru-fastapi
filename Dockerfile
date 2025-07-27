FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y wget unzip libaio1 build-essential

# Oracle Instant Client 복사 및 설치
COPY instantclient-basic-linux.x64-23.8.0.25.04.zip .
RUN unzip instantclient-basic-linux.x64-23.8.0.25.04.zip -d /opt/oracle && \
    ln -s /opt/oracle/instantclient_* /opt/oracle/instantclient && \
    echo "/opt/oracle/instantclient" > /etc/ld.so.conf.d/oracle-instantclient.conf && \
    ldconfig

# 환경 변수
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient

# 앱 디렉토리 및 의존성
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 포트 및 실행
EXPOSE 8000
CMD ["uvicorn", "course_recommender:app", "--host", "0.0.0.0", "--port", "8000"]
