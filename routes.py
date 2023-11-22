from flask import Flask, request
from model import CosineSimilarityRecommender
dataframe_path = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"

global model
model = CosineSimilarityRecommender()
model.fit(dataframe_path)
model.train()

app = Flask(__name__)

@app.route('/api/recomendar', methods=['POST'])
def recomendar_canciones():
    datos_solicitud = request.json
    cancion_id = datos_solicitud['cancion_id']
    n_canciones = datos_solicitud['n_canciones']
    recomendaciones = model.recomend(cancion_id, n_canciones)
    return recomendaciones.to_json(orient='records')


if __name__ == '__main__':
    app.run(port=5001)  # Ajusta el puerto según tus necesidades
