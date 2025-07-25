
def fetch_emotion_data(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT EMOTION_ID,
               EMOTION_NAME,
               DESCRIPTION
          FROM EMOTION
    """)

    emotion_data = []
    emotions = []

    for EMOTION_ID,EMOTION_NAME, DESCRIPTION in cur:
        print(EMOTION_NAME,DESCRIPTION)
        emotion_data.append(EMOTION_NAME + " : " + DESCRIPTION)
        emotions.append(EMOTION_NAME)

    cur.close()
    return emotion_data, emotions

def fetch_question_data(conn):
    cur = conn.cursor()

    cur.execute("""
        SELECT QUESTION_ID,
               EMBEDDING_TEXT
          FROM QUESTION
    """)

    question_data = []
    for QUESTION_ID,EMBEDDING_TEXT in cur:
        question_data.append(EMBEDDING_TEXT)

    cur.close()

    return question_data