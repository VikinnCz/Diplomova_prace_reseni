import os

from beat_tracking import BeatTrackingLibrosa

def Code_Generation(beats):
    WHOLE = 4
    HALF = 1
    QUOTER = 0
    timeline_animations = []
    i = 0
    for beat in beats:
        if i < 2:
            i += 1
            
        else:
            i = 0
            timeline_animations.append(f"addDrawing({beat:.2f}s, {0.5}s, animPlasmaShot(0.5s, #ffff00, 25%));")

    return timeline_animations



if __name__ == "__main__":
    file_name = "Referencni_skladby/Imanbek & BYOR - Belly Dancer (Official Music Video).wav"

    os.system("cls")

    beat_tracking = BeatTrackingLibrosa(file_name=file_name)

    beats = beat_tracking.Get_beats()
    timeline_animations = Code_Generation(beats=beats)
    spectoda_code = ''.join(timeline_animations)
    print(spectoda_code)




