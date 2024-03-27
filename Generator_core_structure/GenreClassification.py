import librosa
import numpy as np
import tensorflow as tf


class GenreClassification:
    """
    A class for music genre classification.

    This class is designed to classify the genre of a music song and provides the probability of the song belonging to each genre. It utilizes a trained neural network model to compute the probabilities based on a chromagram of a 30-second segment of the analyzed song. The output of the neural network is an array of ten values, where each value represents the probability that the song belongs to a specific genre. The genres for which the network is trained include: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.

    Atributes
    ----------
    audio_path : str
        Address for localization of audio file to analyze.
    genre_predictions : ndarray
        Array of predicted values. Each value belongs to the unique genre.

    Propertys
    ----------
    genres_predictions : ndarray
       Array of predicted values. Each value belongs to the unique genre.
    genre : float
        Get genre with the highest probability.
    """

    def __init__(self, audio_path):
        """
        Parameters
        ----------
        audio_path : str
            Address for lacalization of audio file to analyze.
        """
        self.__audio_path = audio_path
        self.__Predict_genre()
        
    def __Predict_genre(self):
        """
        Used neural network model to predict genre probability.
        """
        model_path = 'Generator_core_structure/model_muj_savedmodel_googleColab_trained/'

        # Load the model
        model = tf.saved_model.load(model_path)
        model_infer = model.signatures["serving_default"]

        tensor = self.__Data_preparation() #Get data in format which is need for the model
        predictions = model_infer(tensor) # Tensor of data output from the model

        # Output data postprocesing to ndarray
        predictions_tensor = predictions['dense_3']
        genres_predictions = predictions_tensor.numpy()
        self.__genres_predictions = genres_predictions.reshape(-1, 1)

    def __Data_preparation(self):
        """
        This function prepare audio file to the format which is requires the neural network model. Data needs to be mono 25 s long and in sample rate 22050 Hz. Then is calculate mel spectogram. Then the spectogram is converted into a tensor. 
        """
        # Load the audio file.
        y, sr = librosa.load(self.__audio_path, sr=22050, mono=True, duration=25) 
        
        # Calculate melspectogram.
        mel = self.__Calc_melspec(y=y, sr=sr)

        # Convert ndArray to tensors.
        tensor = tf.constant(mel)
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.expand_dims(tensor, axis=-1)

        return tensor

    def __Calc_melspec(self, y, sr):
        """
        This function calcutale mel spectogram based on librosa librare.
        """
        n_mels = 128
        hop_length = 512
        n_fft = 1024

        # Calculate the melspectogram of the provided audio data
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
        return mel
    
    @property
    def genres_predictions(self):
        return self.genres_predictions
    
    @property
    def genre(self):
        genre = np.argmax(self.__genres_predictions)
        return genre
    