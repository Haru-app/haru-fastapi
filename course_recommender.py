from dotenv import load_dotenv
load_dotenv()

import os
import oracledb  # ← 변경됨
import torch
import time
import redis
import random

from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util

from saveToRedis import save_similarity_to_redis
from queryEmotion import fetch_emotion_data, fetch_question_data
from calculateSimilarity import calculate_similarity_map

app = FastAPI()
model = None
r = None

@app.on_event("startup")
def load_model():
    global model
    global r
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    r = redis.Redis(host='34.64.210.133', port=6379, decode_responses=True)

# DB 연결 정보
ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_HOST = os.getenv("ORACLE_HOST")
ORACLE_PORT = os.getenv("ORACLE_PORT")
ORACLE_SERVICE_NAME = os.getenv("ORACLE_SERVICE_NAME")
DSN = f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={ORACLE_HOST})(PORT={ORACLE_PORT}))(CONNECT_DATA=(SERVICE_NAME={ORACLE_SERVICE_NAME})))"

def getConn():
    return oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=DSN)

# ↓ 아래는 동일 (fetch_store_data, recommend, create_map 등)

# 3. 매장 데이터 조회
def fetch_store_data():
    conn = getConn()
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
def recommend(    
    emotion_input: Optional[str] = Query(None),
    weather_input: Optional[str] = Query(None)):

    if not emotion_input:
        return JSONResponse(
            status_code=400,
            content={"error": "emotion_input은 필수입니다"}
        )
    if not weather_input:
        return JSONResponse(
            status_code=400,
            content={"error": "weather_input은 필수입니다"}
        )

    start_time = time.time()

    # DB fetch time: 약 1.2
    stores = fetch_store_data()

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

    # 음식점 중 유사도 Top 3 중 랜덤 1개
    food_scores = [(i, float(cosine_scores[0][i])) for i in food_indices]
    food_scores.sort(key=lambda x: x[1], reverse=True)
    top_food_candidates = food_scores[:3]
    top_food = [random.choice(top_food_candidates)] if top_food_candidates else []

    # 카페/베이커리 중 유사도 Top 3 중 랜덤 1개
    cafe_scores = [(i, float(cosine_scores[0][i])) for i in cafe_indices]
    cafe_scores.sort(key=lambda x: x[1], reverse=True)
    top_cafe_candidates = cafe_scores[:3]
    top_cafe = [random.choice(top_cafe_candidates)] if top_cafe_candidates else []

    # 비음식점 유사도 Top 15 중 랜덤 4개 (카테고리당 최대 2개 제한)
    non_food_scores = [(i, float(cosine_scores[0][i])) for i in non_food_indices]
    non_food_scores.sort(key=lambda x: x[1], reverse=True)
    top_non_food_candidates = non_food_scores[:15]
    random.shuffle(top_non_food_candidates)

    top_non_food = []
    category_counts = {}
    max_per_category = 2
    max_total = 4

    for idx, score in top_non_food_candidates:
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


@app.get("/question/similarity-map")
def create_map():
    conn = getConn()
    #todo 감정과 값 업데이트

    emotion_data,emotions = fetch_emotion_data(conn)
    question_data = fetch_question_data(conn)
    
    print(emotion_data)
    print(question_data)

    similarity_map = calculate_similarity_map(emotions,emotion_data,question_data,model)
    
    save_similarity_to_redis(similarity_map,r)
    
    conn.close()
    return {"message": "✅ Redis 저장 완료"}
