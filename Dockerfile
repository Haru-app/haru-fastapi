FROM python:3.11-slim

# 시스템 패키지 설치 및 Oracle Instant Client 다운로드
RUN apt-get update && apt-get install -y wget unzip libaio1 build-essential && \
    mkdir -p /opt/oracle && \
    wget https://download.oracle.com/otn_software/linux/instantclient/instantclient-basiclite-linux.x64-21.1.0.0.0.zip && \
    unzip instantclient-basiclite-linux.x64-21.1.0.0.0.zip -d /opt/oracle && \
    rm instantclient-basiclite-linux.x64-21.1.0.0.0.zip && \
    ln -s /opt/oracle/instantclient_* /opt/oracle/instantclient && \
    echo "/opt/oracle/instantclient" > /etc/ld.so.conf.d/oracle-instantclient.conf && \
    ldconfig

# 환경 변수 설정 (cx_Oracle에서 필요)
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient

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
