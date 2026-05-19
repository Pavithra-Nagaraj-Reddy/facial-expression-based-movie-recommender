from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import base64
import cv2
import numpy as np
import requests

app = Flask(__name__)

API_KEY = "fc0738f4520b69e4f88f8b1ff50e6316"

# =========================
# EMOTION → GENRE
# =========================
emotion_genres = {
    "happy": 35,
    "sad": 18,
    "angry": 28,
    "fear": 27,
    "surprise": 878,
    "neutral": 12
}

# =========================
# GET TMDB TRAILER (OFFICIAL)
# =========================
def get_tmdb_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
        data = requests.get(url).json()

        if "results" in data:
            for v in data["results"]:
                if v["site"] == "YouTube" and v["type"] == "Trailer":
                    return f"https://www.youtube.com/embed/{v['key']}"
    except:
        pass

    return None


# =========================
# FETCH MOVIES
# =========================
def get_movies_by_genre(genre_id, language):

    url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_genres={genre_id}&with_original_language={language}&sort_by=popularity.desc"

    response = requests.get(url).json()

    movies = []

    if "results" not in response:
        return movies

    for movie in response["results"]:

        if len(movies) >= 12:
            break

        if movie.get("vote_count", 0) < 10:
            continue

        movie_id = movie.get("id")
        movie_name = movie.get("title", "Unknown")

        poster = movie.get("poster_path")
        if not poster:
            continue

        poster_url = "https://image.tmdb.org/t/p/w500" + poster

        # ⭐ DEFAULT PLATFORM ALWAYS (FIX)
        platforms = {
            "YouTube": f"https://www.youtube.com/results?search_query={movie_name}+trailer"
        }

        # TRY OTT DATA
        try:
            provider_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"
            providers_data = requests.get(provider_url, timeout=3).json()

            if "results" in providers_data and "IN" in providers_data["results"]:

                india = providers_data["results"]["IN"]

                if "flatrate" in india:

                    for p in india["flatrate"]:

                        name = p.get("provider_name", "")

                        if "Netflix" in name:
                            platforms["Netflix"] = f"https://www.netflix.com/search?q={movie_name}"

                        elif "Amazon" in name or "Prime" in name:
                            platforms["Prime Video"] = f"https://www.primevideo.com/search/ref=atv_nb_sr?phrase={movie_name}"

                        elif "Hotstar" in name or "Disney" in name:
                            platforms["Hotstar"] = f"https://www.hotstar.com/in/search?q={movie_name}"

        except:
            pass

        movies.append({
            "id": movie_id,
            "name": movie_name,
            "img": poster_url,
            "rating": movie.get("vote_average", 0),
            "trailer": get_tmdb_trailer(movie_id),
            "platforms": platforms
        })

    return movies


# =========================
# HOME
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# DETECT EMOTION
# =========================
@app.route('/detect', methods=['POST'])
def detect():

    data = request.json
    frames = data['frames']
    language = data['language']

    emotion_count = {}

    for frame in frames:

        img_data = base64.b64decode(frame.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = result[0]['dominant_emotion']

        emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

    emotion = max(emotion_count, key=emotion_count.get)

    genre_id = emotion_genres.get(emotion, 12)

    movies = get_movies_by_genre(genre_id, language)

    return jsonify({
        "emotion": emotion,
        "movies": movies
    })


if __name__ == '__main__':
    app.run(debug=True)