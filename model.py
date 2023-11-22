import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class CosineSimilarityRecommender:
    def __init__(self) -> None:
        self._model = None
        self._dataframe = None

    def fit(self, path):
        # Carga el archivo CSV en un DataFrame de pandas
        self._dataframe = pd.read_csv(path)

    def train(self):
        #Feature selection
        features = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

        # Normalize features
        scaler = MinMaxScaler()
        self._dataframe[features] = scaler.fit_transform(self._dataframe[features])

        # Compute cosine similarity between tracks based on selected features
        track_features = self._dataframe[features]
        track_similarity_matrix = cosine_similarity(track_features, track_features)

        self._model = pd.DataFrame(track_similarity_matrix, index=self._dataframe['track_id'], columns=self._dataframe['track_id'])

    def recomend(self, song_id, n_songs):
        chosen_song_index = self._dataframe[self._dataframe['track_id'] == song_id].index[0]
        similarity_scores = self._model.iloc[chosen_song_index]
        similar_songs = similarity_scores.sort_values(ascending=False)
        top_similar_songs = similar_songs.iloc[1:n_songs+1]
        top_similar_song_ids = top_similar_songs.index.tolist()
        top_similar_songs_details = self._dataframe[self._dataframe['track_id'].isin(top_similar_song_ids)]

        return top_similar_songs_details
