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

# 감정 → 키워드 확장 매핑 딕셔너리
EMOTION_TO_KEYWORDS = {
    "기쁨": ["기쁨", "행복", "즐거움", "즐거운", "유쾌함", "명품", "럭셔리", "만족감"], # 0.56
    "여유": ["여유", "편안함", "안정감", "차분함", "휴식", "조용한 공간", "정갈한 맛", "건강한", "워치"], # 0.6
    "설렘": ["설렘", "기대감", "새로운", "감각적", "로맨스", "로맨틱", "란제리", "특별한", "와인", "데이트", "고급스러움", "워치"], # 0.63
    "호기심": ["호기심", "독특한", "이색적인", "창의적인", "체험", "전시", "컬처", "팝업", "개성"], # 0.73
    "무기력": ["위로", "회복", "포근함", "따뜻함", "감성", "힐링", "자연", "정성", "감동", "건강한"], # 0.56
    "스트레스": ["스트레스 해소", "자극", "분출", "강렬함", "활기참", "운동", "스포츠", "액티비티", "명품", "럭셔리", "매콤한 맛", "즐거움", "만족감"], # 0.54
    "활기참": ["활기참", "생동감", "에너지", "활동적", "아웃도어", "운동", "스포츠", "액티비티", "러닝", "모험", "팝업", "영", "캐릭터", "기쁨", "독창성", "혁신"], # 0.61
}


# 날씨 → 키워드 확장 매핑 딕셔너리
WEATHER_TO_KEYWORDS = {
    "맑음": ["화사함", "야외활동", "기분 좋은", "밝은"],
    "흐림": ["차분한", "잔잔한", "실내", "회색빛"],
    "비": ["조용한", "실내 활동", "감성", "우산"],
    "눈": ["포근한", "차분함", "겨울 느낌"]
}


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

# 매장 데이터 조회
def fetch_store_data():
    conn = getConn()
    cur = conn.cursor()
    cur.execute("""
        SELECT store_id,
               store_name,
               category,
               floor,
               description,
               hashtag,
               image
          FROM store
    """)
    store_data = []
    for store_id, name, category, floor, desc, hashtag_str, image in cur:
        tags = [h.strip() for h in hashtag_str.split(",") if h.strip()]
        store_data.append({
            "store_id": store_id,
            "name": name,
            "category": category,
            "floor": floor,
            "description": desc,
            "tags": tags,
            "image": image
        })
    cur.close()
    conn.close()
    return store_data

# 확장 감정 벡터 생성
# def get_expanded_emotion_vector(emotion: str):
#     keywords = EMOTION_TO_KEYWORDS.get(emotion, [emotion])  # 매핑 없으면 감정 단어 그대로
#     keyword_vecs = model.encode(keywords, convert_to_tensor=True)
#     return keyword_vecs.mean(dim=0)
def get_expanded_emotion_vector(emotion: str):
    keywords = EMOTION_TO_KEYWORDS.get(emotion, [emotion])
    combined = " ".join(keywords)
    return model.encode(combined, convert_to_tensor=True)

# 확장 날짜 벡터 생성
def get_expanded_weather_vector(weather: str):
    keywords = WEATHER_TO_KEYWORDS.get(weather, [weather])
    vecs = model.encode(keywords, convert_to_tensor=True)
    return vecs.mean(dim=0)

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

    # 1. DB에서 매장 데이터 조회
    stores = fetch_store_data()

    # 2. 사용자 임베딩 생성 (감정 0.8 + 날씨 0.2)
    # 확장 감정 벡터 사용
    emotion_vec = get_expanded_emotion_vector(emotion_input)
    weather_vec = get_expanded_weather_vector(weather_input)
    user_embedding = 0.8 * emotion_vec + 0.2 * weather_vec

    # 3. 매장 임베딩 로드 or 계산
    embedding_path = os.getenv("EMBEDDING_PATH", "store_embeddings.pt")
    if os.path.exists(embedding_path):
        store_embeddings = torch.load(embedding_path)
        print("캐시된 store_embeddings 로드")
    else:
        store_inputs = [" ".join(s["tags"]) for s in stores]
        store_embeddings = model.encode(store_inputs, convert_to_tensor=True)
        torch.save(store_embeddings, embedding_path)
        print("store_embeddings 계산 후 저장")

    # 4. 유사도 계산
    cosine_scores = util.cos_sim(user_embedding, store_embeddings)

    # 5. 음식점/카페/비음식점 분리
    # 음식점 카테고리
    food_categories = ['푸드스트리트/델리파크', '22 푸드트럭 피아자', '전문식당가']
    # 카페/베이커리 카테고리
    cafe_categories = ['F&B', '카페 · 베이커리 · 디저트']
    # 음식점과 비음식점 인덱스 나누기
    food_indices = [i for i, s in enumerate(stores) if s['category'] in food_categories]
    cafe_indices = [i for i, s in enumerate(stores) if s['category'] in cafe_categories]
    non_food_indices = [i for i in range(len(stores)) if i not in food_indices]

    # 6. 음식점 top3 중 랜덤 1개
    food_scores = [(i, float(cosine_scores[0][i])) for i in food_indices]
    food_scores.sort(key=lambda x: x[1], reverse=True)
    top_food_candidates = food_scores[:3]
    top_food = [random.choice(top_food_candidates)] if top_food_candidates else []

    # 7. 카페 top3 중 랜덤 1개
    cafe_scores = [(i, float(cosine_scores[0][i])) for i in cafe_indices]
    cafe_scores.sort(key=lambda x: x[1], reverse=True)
    top_cafe_candidates = cafe_scores[:3]
    top_cafe = [random.choice(top_cafe_candidates)] if top_cafe_candidates else []

    # 8. 비음식점 top15 중 랜덤 4개 (카테고리별 최대 2개)
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

    # 9. 추천 매장 결과 리스트 생성
    top_indices = top_food + top_cafe + top_non_food 
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
            'image': store['image'],
            'score': score
        })
    
    # 10. 최종 추천된 코스 유사도 평균 점수
    if recommended_stores:
        average_score = sum(s['score'] for s in recommended_stores) / len(recommended_stores)
    else:
        average_score = 0.0

    # 11. 최대 유사도 기준 평균 점수 (top 6개 전체 중)
    all_scores = [float(score) for score in cosine_scores[0]]
    top6_scores = sorted(all_scores, reverse=True)[:6]
    max_possible_score = sum(top6_scores) / 6 if top6_scores else 0.0

    # 12. 최종 추천된 카테고리 다양성 점수 (최대 6개 카테고리)
    category_set = set([s['category'] for s in recommended_stores])
    diversity_score = len(category_set) / 6

    # 13. 최종 추천된 코스 유사도 표준편차 (분산이 작을수록 고르게 유사)
    import statistics
    score_list = [s['score'] for s in recommended_stores]
    score_std = statistics.stdev(score_list) if len(score_list) > 1 else 0.0

    # 14. 최종 결과 리스트 층수 기준으로 정렬
    def parse_floor(floor_str):
        if '지하' in floor_str:
            num = ''.join(filter(str.isdigit, floor_str))
            return -int(num) if num else -100
        else:
            num = ''.join(filter(str.isdigit, floor_str))
            return int(num) if num else 100

    sorted_stores = sorted(recommended_stores, key=lambda x: parse_floor(x['floor']))

    # 디버깅 로그 출력
    print("추천 매장 (층순 정렬):")
    for store in sorted_stores:
        print(f"매장ID: {store['store_id']}, 매장명: {store['name']}, 카테고리: {store['category']}, 층: {store['floor']}, 설명: {store['description']}, 해시태그: {store['tags']}, 유사도: {store['score']:.4f}")

    end_time = time.time()
    print(f"걸린 시간: {end_time - start_time:.2f}초")

    # 15. 최종 응답
    return JSONResponse(content={
        "stores": sorted_stores,
        "average_score": round(average_score, 4),
        "max_possible_score": round(max_possible_score, 4),
        "diversity_score": round(diversity_score, 4),
        "similarity_stddev": round(score_std, 4)
    })
    # return JSONResponse(content={"stores": sorted_stores})


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
