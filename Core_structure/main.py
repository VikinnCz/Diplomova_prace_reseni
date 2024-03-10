import os
import numpy as np

from BeatTracking import BeatTracking
from ChromaFeatures import ChromaFeatures

CHILL       = 0
HANG_OUT    = 1
HAPPY       = 2
DANCING     = 3

def Code_Generation(beat_tracking:BeatTracking, chroma_features:ChromaFeatures):
    WHOLE = 4
    HALF = 1
    QUOTER = 0

    beats = beat_tracking.Get_beats()
    beats_strength = beat_tracking.Get_strength()
    times = beat_tracking.Get_times()
    chroma = chroma_features.Get_chroma()
    tones_colors = chroma_features.Get_tones_colors()

    beats_stength_average = np.median(beats_strength)

    timeline_animations = []

    for index, beat in enumerate(beats):

        if beats_strength[index] > (beats_stength_average-0.2):
            range = 50
            
            time_index =np.where(times == beat)[0]
            if time_index - range >= 0 and time_index + range < chroma.shape[1]:
                chroma_in_range = np.sum(chroma[:,int(time_index-range):int(time_index+range)], axis=1)
                primary_tone = chroma_in_range.argmax()
            else:
                primary_tone = chroma[:,time_index].argmax() # Tohle je funkcni bez casoveho rozmezi
             

            tone_color = tones_colors[primary_tone]
            hex_tone_color = '#%02x%02x%02x' % (tone_color[0], tone_color[1], tone_color[2])

            timeline_animations.append(f"addDrawing({beat:.2f}s, 0.5s, animPlasmaShot(0.5s, {hex_tone_color}, 25%));")

    print (len(timeline_animations))
    return timeline_animations



if __name__ == "__main__":
    file_name = "Referencni_skladby/Imanbek & BYOR - Belly Dancer (Official Music Video).wav"
    # file_name = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/07.-Here Comes The Sun.wav"
    # file_name = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/01.-Come Together.wav"
    

    os.system("cls")


    beat_tracking = BeatTracking(file_name=file_name)
    chroma_features = ChromaFeatures(file_name=file_name, mood=HAPPY)

    # Na základě mood můžu nastavovat trashold pro beat strenght

    timeline_animations = Code_Generation(beat_tracking, chroma_features)
    spectoda_code = ''.join(timeline_animations)
    print(spectoda_code)




