import librosa
import Constants

from ChromaFeatures import ChromaFeatures

class Segmentation:
    """ S class for audio segmentation.

    Thsi class analyse music song and divides it into similar segments number of computed segments are variable. 

    Atributes
    ----------
    chroma_features : ChromaFeatures
        Is class object that provides important information abou chrom structure of audio.
    mood : int
        Mood of the user.

    Propertys
    ----------
    bounds : ndarray
        Array of times where the boundaries of segments are placed.


    """

    def __init__(self, chroma_features : ChromaFeatures, mood : int):
        """
        Parameters
        ----------
        chroma_features : ChromaFeatures
            Is class object that provides important information about chrom structure of audio.
        """
        self.__chroma_features = chroma_features
        self.__mood = mood
        self.__Calc_segments()

    def __Segments_number(self):
        match self.__mood:
            case Constants.CHILL:
                return 8
            case Constants.HANG_OUT:
                return 16
            case Constants.HAPPY: 
                return 24
            case Constants.DANCING:
                return 32
            case _:
                return 16

    def __Calc_segments(self,):
        """
        Calculate similarities in audio chroma features and compute segments of similar features in audio. Save timestamps of bounds of computed segments.
        """
        chroma = self.__chroma_features.crhoma
        sr = self.__chroma_features.sr
        bounds = librosa.segment.agglomerative(chroma, self.__Segments_number())
        self.__bounds = librosa.frames_to_time(bounds, sr=sr)

    @property
    def bounds(self):
        """Get timestamps of segments bounds"""
        return self.__bounds