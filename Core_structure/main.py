import os
import numpy as np

from BeatTracking import BeatTracking

def Code_Generation(beats, beats_stength):
    WHOLE = 4
    HALF = 1
    QUOTER = 0

    beats_stength_average = np.median(beats_stength)

    timeline_animations = []
    # i = 0
    for index, beat in enumerate(beats):
        # if i < HALF:
        #     i += 1
        if beats_stength[index] > beats_stength_average:
            timeline_animations.append(f"addDrawing({beat:.2f}s, 0.5s, animPlasmaShot(0.5s, #ffff00, 25%));")
        # else:
        #     i = 0
        # timeline_animations.append(f"addDrawing({beat:.2f}s, 0.5s, animPlasmaShot(0.5s, #ffff00, 25%));")
    print (len(timeline_animations))
    return timeline_animations



if __name__ == "__main__":
    file_name = "Referencni_skladby/Imanbek & BYOR - Belly Dancer (Official Music Video).wav"
    # file_name = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/07.-Here Comes The Sun.wav"
    # file_name = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/01.-Come Together.wav"
    

    os.system("cls")

    BeatTracking = BeatTracking(file_name=file_name)

    beats = BeatTracking.Get_beats()
    beats_stength = BeatTracking.Get_strength()

    timeline_animations = Code_Generation(beats=beats, beats_stength=beats_stength)
    spectoda_code = ''.join(timeline_animations)
    print(spectoda_code)




