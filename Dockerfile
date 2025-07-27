FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y wget unzip libaio1 build-essential && \
    rm -rf /var/lib/apt/lists/*

# Oracle Instant Client 다운로드 및 설치
RUN mkdir -p /opt/oracle && \
    wget https://download.oracle.com/otn_software/linux/instantclient/2380000/instantclient-basic-linux.x64-23.8.0.25.04.zip && \
    unzip instantclient-basic-linux.x64-23.8.0.25.04.zip -d /opt/oracle && \
    rm instantclient-basic-linux.x64-23.8.0.25.04.zip && \
    ln -s /opt/oracle/instantclient_23_8 /opt/oracle/instantclient && \
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
