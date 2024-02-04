import librosa
import numpy as np

class BeatTracking:
    def __init__(self, file_name):
        self.__Calc_beats(file_name)

    def __Calc_beats(self, file_name):
        y, sr = librosa.load(file_name)
        self.tempo, self.__beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        times = librosa.times_like(onset_env, sr=sr, hop_length=512)
        self.__beats = times[self.__beats]
        
        self.__Calc_strength(times, onset_env)
    
    def __Calc_strength(self, times, onset_env):
        self.__strength = np.ones(len(self.__beats))
        i = 0
        for beat in self.__beats:
            try:
                index = np.where(times == beat)[0]
                self.__strength[i] = self.__Max_of_range(int(index), onset_env)
            except ValueError:
                self.__strength[i] = 0
            i += 1

        self.__strength = librosa.util.normalize(self.__strength)
    
    def __Max_of_range (self, x, onset_env):
        range_size  = 3
        start = max(x - range_size, 0)
        end = min(x + range_size + 1, len(onset_env))
        range = onset_env[start:end]
        return np.max(range)
    
    def Get_beats(self):
        return self.__beats
    def Get_strength(self):
        return self.__strength
    def Get_tempo(self):
        return self.__tempo
    

