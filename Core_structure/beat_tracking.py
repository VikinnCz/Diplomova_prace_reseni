import librosa
import numpy as np

class BeatTrackingLibrosa:
    def __init__(self, file_name):
        self.file_name = file_name

    def Get_beats(self):
        y, sr = librosa.load(self.file_name)
        tempo_librosa, beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        times = librosa.times_like(onset_env, sr=sr, hop_length=512)
        beats = times[beats]
        return beats
