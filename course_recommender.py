from dotenv import load_dotenv
load_dotenv()

import os
import cx_Oracle
import torch
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util

# 앱 시작 시 모델 로딩
app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # model time: 약 3.6

# 1. Oracle Instant Client 초기화
lib_dir = os.path.join(os.path.dirname(__file__), os.getenv("INSTANT_CLIENT_DIR"))
cx_Oracle.init_oracle_client(lib_dir=lib_dir)

# 2. DB 연결 정보
ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_HOST = os.getenv("ORACLE_HOST")
ORACLE_PORT = os.getenv("ORACLE_PORT")
ORACLE_SERVICE_NAME = os.getenv("ORACLE_SERVICE_NAME")
DSN = cx_Oracle.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE_NAME)

# 3. 매장 데이터 조회
def fetch_store_data():
    conn = cx_Oracle.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=DSN)
    cur = conn.cursor()
    cur.execute("""
        SELECT store_id,
               store_name,
               category,
               floor,
               description,
               hashtag
          FROM store
    """)
    store_data = []
    for store_id, name, category, floor, desc, hashtag_str in cur:
        tags = [h.strip() for h in hashtag_str.split(",") if h.strip()]
        store_data.append({
            "store_id": store_id,
            "name": name,
            "category": category,
            "floor": floor,
            "description": desc,
            "tags": tags
        })
    cur.close()
    conn.close()
    return store_data

@app.get("/recommend")
def recommend(emotion_input1: str, emotion_input2: str, weather_input: str):
    start_time = time.time()

    # DB fetch time: 약 1.2
    stores = fetch_store_data()

    # 사용자 입력을 하나의 문장으로 합침 (감정 + 날씨)
    # user_input = f"{emotion_input1} {emotion_input2} {weather_input}"
    # 4. 사용자 입력 임베딩 소요 시각 약 0.07
    #user_embedding = model.encode(user_input, convert_to_tensor=True)

    emotion_input = f"{emotion_input1} {emotion_input2}"
    weather_input = weather_input

    emotion_vec = model.encode(emotion_input, convert_to_tensor=True)
    weather_vec = model.encode(weather_input, convert_to_tensor=True)

    user_embedding = 0.7 * emotion_vec + 0.3 * weather_vec

    # 5. 매장 임베딩 (일단 변경사항 없을 것이라고 예상)
    embedding_path = os.getenv("EMBEDDING_PATH", "store_embeddings.pt")
    if os.path.exists(embedding_path):
        store_embeddings = torch.load(embedding_path)  # load time: 약 0.001
        print("캐시된 store_embeddings 로드")
    else:
        store_inputs = [" ".join(s["tags"]) for s in stores]
        store_embeddings = model.encode(store_inputs, convert_to_tensor=True)
        torch.save(store_embeddings, embedding_path)
        print("store_embeddings 계산 후 저장")

    # 6. 코사인 유사도 계산
    cosine_scores = util.cos_sim(user_embedding, store_embeddings)

    # 음식점 카테고리
    food_categories = ['푸드스트리트/델리파크', '22 푸드트럭 피아자', '전문식당가']

    # 카페/베이커리 카테고리
    cafe_categories = ['F&B', '카페 · 베이커리 · 디저트']

    # 음식점과 비음식점 인덱스 나누기
    food_indices = [i for i, s in enumerate(stores) if s['category'] in food_categories]
    cafe_indices = [i for i, s in enumerate(stores) if s['category'] in cafe_categories]
    non_food_indices = [i for i in range(len(stores)) if i not in food_indices]

    # 음식점 중 유사도 가장 높은 1개
    food_scores = [(i, float(cosine_scores[0][i])) for i in food_indices]
    food_scores.sort(key=lambda x: x[1], reverse=True)
    top_food = food_scores[:1]

    # 카페/베이커리 중 유사도 가장 높은 1개
    cafe_scores = [(i, float(cosine_scores[0][i])) for i in cafe_indices]
    cafe_scores.sort(key=lambda x: x[1], reverse=True)
    top_cafe = cafe_scores[:1]

    # 비음식점 중 카테고리별 2개 이하, 총 4개 선택
    non_food_scores = [(i, float(cosine_scores[0][i])) for i in non_food_indices]
    non_food_scores.sort(key=lambda x: x[1], reverse=True)
    top_non_food = []
    category_counts = {}
    max_per_category = 2
    max_total = 4

    for idx, score in non_food_scores:
        category = stores[idx]['category']
        if category_counts.get(category, 0) < max_per_category:
            top_non_food.append((idx, score))
            category_counts[category] = category_counts.get(category, 0) + 1
        if len(top_non_food) >= max_total:
            break

    # 추천 인덱스 통합
    top_indices = top_food + top_cafe + top_non_food 

    # 7. 추천 결과 리스트 생성
    recommended_stores = []
    for idx, score in top_indices:
        store = stores[idx]
        recommended_stores.append({
            'store_id': store['store_id'],
            'name': store['name'],
            'category': store['category'],
            'floor': store['floor'],
            'description': store['description'],
            'tags': store['tags'],
            'score': score
        })

    # 층수 기준으로 정렬
    def parse_floor(floor_str):
        if '지하' in floor_str:
            num = ''.join(filter(str.isdigit, floor_str))
            return -int(num) if num else -100
        else:
            num = ''.join(filter(str.isdigit, floor_str))
            return int(num) if num else 100

    sorted_stores = sorted(recommended_stores, key=lambda x: parse_floor(x['floor']))

    # 8. 정렬된 결과 출력
    print("추천 매장 (층순 정렬):")
    for store in sorted_stores:
        print(f"매장ID: {store['store_id']}, 매장명: {store['name']}, 카테고리: {store['category']}, 층: {store['floor']}, 설명: {store['description']}, 해시태그: {store['tags']}, 유사도: {store['score']:.4f}")

    end_time = time.time()
    print(f"걸린 시간: {end_time - start_time:.2f}초")

    return JSONResponse(content={"stores": sorted_stores})
