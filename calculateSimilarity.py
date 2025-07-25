import json
import numpy as np
from sentence_transformers import util

def emphasize_similarity(sim_list, weight=5.0):
    sim_arr = np.array(sim_list)
    exp_scaled = np.exp(sim_arr * weight)
    return (exp_scaled / exp_scaled.sum()).round(4).tolist()


def calculate_similarity_map(emotions,emotion_descriptions,questions,model):
    # 임베딩
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    emotion_embeddings = model.encode(emotion_descriptions, convert_to_tensor=True)

    # 유사도 계산 및 저장
    similarity_map = {}

    for i, question in enumerate(questions):
        qid = f"Q{i+1}"
        sims = []
        for j, emotion in enumerate(emotions):
            sim = util.cos_sim(question_embeddings[i], emotion_embeddings[j]).item()
            sims.append(sim)


        emphasized = emphasize_similarity(sims, weight=5.0)  # 여기만 바꿨음
        similarity_map[qid] = {
            emotions[j]: round(score, 4)
            for j, score in enumerate(emphasized)
        }
    
    return similarity_map