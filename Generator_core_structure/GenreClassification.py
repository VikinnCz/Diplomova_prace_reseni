import librosa
import numpy as np
import tensorflow as tf


class GenreClassification:

    def __init__(self, audio_path):

        self.__audio_path = audio_path
        self.__Predict_genre()
        


    def __Predict_genre(self):
        model_path = 'Generator_core_structure/model_muj_savedmodel_googleColab_trained/'

        # Načtení modelu
        model = tf.saved_model.load(model_path)
        model_infer = model.signatures["serving_default"]

        tensor = self.__Data_preparation()
        predictions = model_infer(tensor)

        predictions_tensor = predictions['dense_3']
        genres_predictions = predictions_tensor.numpy()
        self.__genres_predictions = genres_predictions.reshape(-1, 1)

    def __Data_preparation(self):
        y, sr = librosa.load(self.__audio_path, sr=22050, mono=True, duration=25)

        mel = self.__Calc_melspec(y=y, sr=sr)
        tensor = tf.constant(mel)
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.expand_dims(tensor, axis=-1)

        return tensor

    def __Calc_melspec(self, y, sr):
        n_mels = 128
        hop_length = 512
        n_fft = 1024
        #calculate the melspectogram of the provided audio wave
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
        return mel
    
    def Get_genres_predictions(self):
        return self.__genres_predictions
    
    def Get_genre(self):
        genre = np.argmax(self.__genres_predictions)
        return genre
    