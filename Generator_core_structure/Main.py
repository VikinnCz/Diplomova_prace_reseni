import os
import json
import Constants
import numpy as np

from Dataset import Dataset
from Segmentation import Segmentation
from BeatTracking import BeatTracking
from AnimationBlock import AnimationBlock
from ChromaFeatures import ChromaFeatures
from GenreClassification import GenreClassification

def Code_generation(beat_tracking:BeatTracking, chroma_features:ChromaFeatures):
    WHOLE = 4
    HALF = 1
    QUOTER = 0

    beats = beat_tracking.beats
    beats_strength = beat_tracking.strength
    times = beat_tracking.times
    chroma = chroma_features.crhoma
    tones_colors = chroma_features.tones_colors

    beats_stength_median = np.median(beats_strength)

    timeline_animations = []

    for index, beat in enumerate(beats):

        if beats_strength[index] > (beats_stength_median-0.2):
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

    os.system("cls")
    print (len(timeline_animations)) #Počet animací
    return timeline_animations

def Dataset_selection(dataset_database : list[Dataset], genre_classification : GenreClassification, beat_tracking : BeatTracking, mood : int):

    genre = genre_classification.genre
    tempo = beat_tracking.tempo

    selected_dataset = Dataset
    this_tempo_difrence = int

    for dataset in dataset_database:
        if dataset.genre == genre and dataset.mood_characteristics == mood:
            new_tempo_difrence = abs(dataset.tempo - tempo)
            if this_tempo_difrence > new_tempo_difrence:
                this_tempo_difrence = new_tempo_difrence
                selected_dataset = dataset

    return selected_dataset

def Load_dataset_database():
    with open ("dataset_database.json", "r") as fp:
        dataset_database_json = json.load(fp,)

    dataset_database = [Dataset(**dataset) for dataset in dataset_database_json]

    for dataset in dataset_database:
        print (dataset.to_dict())

    return [Dataset]


if __name__ == "__main__":
    # audio_path = "Referencni_skladby/Imanbek & BYOR - Belly Dancer (Official Music Video).wav"
    audio_path = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/07.-Here Comes The Sun.wav"
    # audio_path = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/01.-Come Together.wav"

    dataset_database = Load_dataset_database()
    mood = Constants.HAPPY

    beat_tracking = BeatTracking(audio_path=audio_path)
    chroma_features = ChromaFeatures(audio_path=audio_path, mood=mood)
    genre_classification = GenreClassification(audio_path=audio_path)
    segmentation = Segmentation(audio_path=audio_path, chroma_features=chroma_features)
    segmentation.bounds
    

    dataset = Dataset_selection(dataset_database, genre_classification, beat_tracking, mood)

    timeline_animations = Code_generation(beat_tracking, chroma_features)
    spectoda_code = ''.join(timeline_animations)
    print(spectoda_code)


## Parametry které je možné nastavovat a na základě toho měnit generování aniací. ##
    
# Na základě mood můžu nastavovat trashold pro beat strenght
# Počet generovaných segmentů (number_of_segments)