import librosa

from ChromaFeatures import ChromaFeatures

class Segmentation:

    def __init__(self, audio_path : str,
                    chroma_features : ChromaFeatures,
                    number_of_segments = 12):
        self.__audio_path = audio_path
        self.__chroma_features = chroma_features
        self.__number_of_segments = number_of_segments
        
        self.__Calc_segments()


    def __Calc_segments(self):
        
        chroma = self.__chroma_features.crhoma
        sr = self.__chroma_features.sr
        bounds = librosa.segment.agglomerative(chroma, self.__number_of_segments)
        self.__bounds = librosa.frames_to_time(bounds, sr=sr)

    @property
    def bounds(self):
        return self.bounds