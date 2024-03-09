import librosa
import numpy as np

class BeatTracking:
    """A class for Beat tracking analysis

    This class analyse rhythmic structure of music song and provides a arrays with beats in times. Also provides strength of the beats.

    Atributes
    ----------
    file_name : str
        Address for lacalization of audio file to analyse.
    beats : ndarray
        Array of obtained beats.
    strength :
        Strength of obtained beats.
    times :
        The times in which the botained beats are located.
    tempo :
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


    def __init__(self, file_name):
        """
        Parameters
        ----------
        file_name : str
            Address for lacalization of audio file to analyse.
        """
        self.__Calc_beats(file_name)

    def __Calc_beats(self, file_name):
        """
        Parameters
        ----------
        file_name : str
            Address for lacalization of audio file to analyse.
        """
        y, sr = librosa.load(file_name)
        self.tempo, self.__beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        self.__times = librosa.times_like(onset_env, sr=sr, hop_length=512)
        self.__beats = self.__times[self.__beats]
        
        self.__Calc_strength(onset_env)
    
    def __Calc_strength(self, onset_env):
        """
        Parameters
        ----------
        onset_env : ndarray
            Onset envelope
        """
        self.__strength = np.ones(len(self.__beats))
        i = 0
        for beat in self.__beats:
            try:
                index = np.where(self.__times == beat)[0]
                self.__strength[i] = self.__Max_of_range(int(index), onset_env)
            except ValueError:
                self.__strength[i] = 0
            i += 1

        self.__strength = librosa.util.normalize(self.__strength)
    
    def __Max_of_range (self, index, onset_env):
        """
        Parameters
        ----------
        index : int
            Index of time value where is the beat located. 
        onset_env : ndarray
            Onset envelope
        """
        range_size  = 3
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
    

