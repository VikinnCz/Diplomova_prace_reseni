import librosa

from ChromaFeatures import ChromaFeatures

class Segmentation:
    """ S class for audio segmentation.

    Thsic class analyse music song and divides it into similar segments number of computed segments are variable. 

    Atributes
    ----------
    audio_path: str
        Address for localization of audio file to analyze.
    chroma_features : ChromaFeatures
        Is class object that provides important information abou chrom structure of audio.
    number_of_segments : int, optional
        This number determines into how many segments the recording is dividede. Default value is 12. 

    Propertys
    ----------
    bounds : ndarray
        Array of times where the boundaries of segments are placed.


    """

    def __init__(self, audio_path : str,
                    chroma_features : ChromaFeatures,
                    number_of_segments = 12):
        """
        Parameters
        ----------
        audio_path: str
            Address for localization of audio file to analyze.
        chroma_features : ChromaFeatures
            Is class object that provides important information abou chrom structure of audio.
        number_of_segments : int, optional
            This number determines into how many segments the recording is dividede. Default value is 12. 
        """
        self.__audio_path = audio_path
        self.__chroma_features = chroma_features
        self.__number_of_segments = number_of_segments
        self.__Calc_segments()


    def __Calc_segments(self):
        """
        Calculate similarities in audoi chroma features and compute segments of similar features in audio. Save timestamps of bounds of computed segments.
        """
        chroma = self.__chroma_features.crhoma
        sr = self.__chroma_features.sr
        bounds = librosa.segment.agglomerative(chroma, self.__number_of_segments)
        self.__bounds = librosa.frames_to_time(bounds, sr=sr)

    @property
    def bounds(self):
        """Get timestamps of segments bounds"""
        return self.__bounds