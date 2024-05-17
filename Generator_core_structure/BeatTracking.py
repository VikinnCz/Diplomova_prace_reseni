import librosa
import numpy as np

class BeatTracking:
    """A class for Beat tracking analysis.

    This class analyze rhythmic structure of music song and provides a arrays with beats in times. Also provides strength of the beats.

    Attributes
    ----------
    y : array
        Samples of audio for analyze.
    sr : float
        Sample rate of audio for analyze.
    beats : array
        Array of obtained beats.
    strength : array
        Strength of obtained beats.
    times : array
        The times in which the obtained beats are located.
    tempo : float
        Tempo of analyzed audio file. In BPM.

    Methods
    ----------
    Get_beats()
        Return array of obtained beats.
    Get_strength()
        Return strength of obtained beats.
    Get_tempo()
        Return tempo of analyzed audio file. In BPM.
    Get_times()
        Return the times in which the obtained beats are located.
    """


    def __init__(self, y, sr):
        """
        Parameters
        ----------
        y : array
            Samples of audio for analyze.
        sr : float
            Sample rate of audio for analyze.
        """
        self.__y = y
        self.__sr = sr
        self.__Calc_beats()

    def __Calc_beats(self):
        """
        Function use the librosa library to analyze the audio file. The output is an array of beats.

        Parameters
        ----------
        y : array
            Samples of audio for analyze.
        sr : float
            Sample rate of audio for analyze.
        """

        self.__tempo, self.__beats = librosa.beat.beat_track(y=self.__y, sr=self.__sr)
        onset_env = librosa.onset.onset_strength(y=self.__y, sr=self.__sr, aggregate=np.median)
        self.__times = librosa.times_like(onset_env, sr=self.__sr, hop_length=512)
        self.__beats = self.__times[self.__beats] # Transform beats into their timestamps
        
        self.__Calc_strength(onset_env)
    
    def __Calc_strength(self, onset_env):
        """
        Calculate strength of beats.

        The function calculate beats strength based on onset envelope in time of the beat. Function also check range around the beat if there is some bigger value in onset envelope.

        Parameters
        ----------
        onset_env : ndarray
            Onset envelope
        """
        self.__strength = np.ones(len(self.__beats)) # Declaration of ones ndarray.
        i = 0

        for beat in self.__beats:
            try:
                index = np.where(self.__times == beat)[0] # Getting a timestamp of the beat.
                self.__strength[i] = self.__Max_of_range(int(index), onset_env) # Gets a biggest onset value in range around the timestamp of beat.
            except ValueError:
                self.__strength[i] = 0
            i += 1

        self.__strength = librosa.util.normalize(self.__strength) # The beat strength normalization between values 0-1.
    
    def __Max_of_range (self, index : int, onset_env, range_size  = 3):
        """
        Get max value in range

        The function return max value of array in range around index.

        Parameters
        ----------
        index : int
            Index of time value where is the beat located. 
        onset_env : ndarray
            Onset envelope
        range_size : int, optional
            The range in which the maximum value is searched for. (default is 3)

        Returns
        ----------
        float
            Maximum value from the array in the range.
        """
        start = max(index - range_size, 0)
        end = min(index + range_size + 1, len(onset_env))
        range = onset_env[start:end]

        return np.max(range)
    
    @property
    def beats(self):
        return self.__beats
    
    @property
    def strength(self):
        return self.__strength
    
    @property
    def tempo(self):
        return self.__tempo
    
    @property
    def times(self):
        return self.__times
    

