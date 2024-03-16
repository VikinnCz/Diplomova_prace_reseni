import librosa
import numpy as np

class BeatTracking:
    """A class for Beat tracking analysis.

    This class analyze rhythmic structure of music song and provides a arrays with beats in times. Also provides strength of the beats.

    Atributes
    ----------
    audio_path : str
        Address for localization of audio file to analyze.
    beats : array
        Array of obtained beats.
    strength : array
        Strength of obtained beats.
    times : array
        The times in which the botained beats are located.
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
        Return the times in which the botained beats are located.
    """


    def __init__(self, audio_path):
        """
        Parameters
        ----------
        audio_path : str
            Address for lacalization of audio file to analyze.
        """
        self.__Calc_beats(audio_path)

    def __Calc_beats(self, audio_path):
        """
        Function use the librosa library to analyze the audio file. The output is an array of beats.

        Parameters
        ----------
        audio_path : str
            Address for lacalization of audio file to analyze.
        """

        y, sr = librosa.load(audio_path)
        self.tempo, self.__beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        self.__times = librosa.times_like(onset_env, sr=sr, hop_length=512)
        self.__beats = self.__times[self.__beats] # Transform beats into their timestamps
        
        self.__Calc_strength(onset_env)
    
    def __Calc_strength(self, onset_env):
        """
        Calculate strenght of beats.

        The function calculate beats strenght based on onset envelope in time of the beat. Function also check range around the beat if there is some bigger value in onset envelope.

        Parameters
        ----------
        onset_env : ndarray
            Onset envelope
        """
        self.__strength = np.ones(len(self.__beats)) # Declrataion of ones ndarray.
        i = 0

        for beat in self.__beats:
            try:
                index = np.where(self.__times == beat)[0] # Geting a timestamp of the beat.
                self.__strength[i] = self.__Max_of_range(int(index), onset_env) # Gets a bigest onset value in range around the timestamp of beat.
            except ValueError:
                self.__strength[i] = 0
            i += 1

        self.__strength = librosa.util.normalize(self.__strength) # The beat strength normalization between values 0-1.
    
    def __Max_of_range (self, index : int, onset_env, range_size  = 3):
        """
        Get max value in range

        The function return max value of array in ragne around index.

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
    
    def Get_beats(self):
        return self.__beats 
    def Get_strength(self):
        return self.__strength
    def Get_tempo(self):
        return self.__tempo
    def Get_times(self):
        return self.__times
    

