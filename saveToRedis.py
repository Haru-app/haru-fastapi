# save_to_redis.py

import json

def save_similarity_to_redis(similarity_map,r):

    # Redis 저장
    r.set("question_emotion_similarity.json", json.dumps(similarity_map, ensure_ascii=False, indent=2), ex=36000)
    print("✅ Redis에 저장 완료!")