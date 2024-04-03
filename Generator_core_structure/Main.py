import os
import json
import Constants
import numpy as np
import librosa
import pyloudnorm as pyln

from Dataset import Dataset
from Segmentation import Segmentation
from BeatTracking import BeatTracking
from AnimationBlock import AnimationBlock
from ChromaFeatures import ChromaFeatures
from GenreClassification import GenreClassification

def Code_generation(y, sr, beat_tracking:BeatTracking, chroma_features:ChromaFeatures):
    WHOLE = 4
    HALF = 1
    QUOTER = 0

    beats = beat_tracking.beats
    beats_strength = beat_tracking.strength
    times = beat_tracking.times
    chroma = chroma_features.crhoma
    tones_colors = chroma_features.tones_colors
    segments = segmentation.bounds

    beats_stength_median = np.median(beats_strength)
    audio_loudness = Calc_loudness(y,sr)

    timeline_animations = []
    completed = False

    start_beat = beats[0]
    while not completed:

        # TODO: Vybrat vhodný blok animace
            # TODO: Segment
                # TODO: Délka segmentu - začátek a konec
                # TODO: O jaký segment se jedná (loudness segmentu v poměru loudness celé skladby)
            # TODO: Beat 
                # TODO: O jaký beat se jedná jeho síla v okolí času. 
                # TODO: Jeho síla v porovnání s mediánem síly ostatních beatů v celé skladbě
                # TODO: Jeho síla v porovnání s mediánem síly ostatních beatů v daném segmentu
            # TODO: Čas k dalšímu podobnému beatu
        
        # V jakém segmentu se nachází start_beat
        start_segment, end_segment = Find_segment(segments, start_beat)
        segment_duration = end_segment - start_beat

        # Hlasitost segmentu
        segment_loudness = Calc_loudness(start_time=start_segment, end_time=end_segment, y=y, sr=sr)



        # TODO: Přiřazení barev

        # TODO: Délka animace

        # TODO: Vygenerování spectoda codu animace.
        # TODO: Vybrat následující start_beat jako end_beat
        start_beat = end_beat

        # TODO: Kontrolní podmínka jestli už je vygenerovaná animace pro celou skladbu







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

def Find_segment(array, value):
    array = np.asarray(array)
    idx = ((array - value)).argmin()

    if array[idx] > value:
        idx -= 1

    return array[idx], array[idx+1]

def Calc_loudness(start_time, end_time, y, sr):
    start_sample = int(sr*start_time)
    end_sample = int(sr*end_time)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y[start_sample:end_sample])
    return loudness

def Calc_loudness(y, sr):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    return loudness





def Dataset_selection(dataset_database : list[Dataset], genre_classification : GenreClassification, beat_tracking : BeatTracking, mood : int):
    
    genre_predictions = genre_classification.genres_predictions
    tempo = beat_tracking.tempo

    genres_difs = []

    # Browsing thru all datasets
    for i, dataset in enumerate(dataset_database):
        genre_dif = 0
        d_genres_prediction = dataset.genre
        for key in d_genres_prediction:
            genre_dif += d_genres_prediction[key] - genre_predictions[key]

        genres_difs.append(np.abs(genre_dif))

    genre_pass_datasets = []

    for i in range(5):
        index_of_min = int(np.argmin(genres_difs))
        genre_pass_datasets.append( dataset_database[index_of_min])
        genres_difs[index_of_min] = 255

    this_tempo_dif = 100
    selected_dataset = Dataset
    
    for dataset in genre_pass_datasets:
        if dataset.mood == mood:
            new_tempo_dif = abs(dataset.tempo - tempo)
            if this_tempo_dif > new_tempo_dif:
                this_tempo_dif = new_tempo_dif
                selected_dataset = dataset


    # ---> Tohle je pro dataset s jednou hodnotou žándru.
    # Broesing thru all datasets
    # for dataset in dataset_database:
    #     if dataset.genre == genre and dataset.mood_characteristics == mood:
    #         new_tempo_difrence = abs(dataset.tempo - tempo)
    #         if this_tempo_difrence > new_tempo_difrence:
    #             this_tempo_difrence = new_tempo_difrence
    #             selected_dataset = dataset

    return selected_dataset

def Load_dataset_database():
    with open ("Generator_core_structure/dataset_database.json", "r") as fp:
        dataset_database_json = json.load(fp,)

    dataset_database = [Dataset(**dataset) for dataset in dataset_database_json]
    return dataset_database


if __name__ == "__main__":
    # audio_path = "Referencni_skladby/Imanbek & BYOR - Belly Dancer (Official Music Video).wav"
    audio_path = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/07.-Here Comes The Sun.wav"
    # audio_path = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/01.-Come Together.wav"

    dataset_database = Load_dataset_database()
    mood = Constants.HAPPY

    y, sr = librosa.load(audio_path)

    # TODO: Předělat všechny třídy s audio_path na (y, sr)
    beat_tracking = BeatTracking(audio_path=audio_path)
    chroma_features = ChromaFeatures(audio_path=audio_path, mood=mood)
    genre_classification = GenreClassification(audio_path=audio_path)
    segmentation = Segmentation(chroma_features=chroma_features)
    segmentation.bounds
    

    dataset = Dataset_selection(dataset_database, genre_classification, beat_tracking, mood)

    timeline_animations = Code_generation(y, sr, beat_tracking, chroma_features)
    spectoda_code = ''.join(timeline_animations)
    print(spectoda_code)


## Parametry které je možné nastavovat a na základě toho měnit generování animací. ##
# Na základě mood můžu nastavovat trashold pro beat strenght
# Počet generovaných segmentů (number_of_segments)


#Jak vybírat datasety abych pro každý žánr nemusel mít dataset pro každou náladu (40 datasetů). 
    # Řešení 1: žánr bude mít nejnižší prioritu při výběru. Prioritní bude mood a tempo až nakonec žánr.
    # Řešení 2: parametr genre bude list a každý dataset může být pro několik žánrů. Například to může být pole s hodnotou pro každý žánr a tato hodnota bude určovat vhodnost pro daný žánr. Tato hodnota se pak porovnává s hodnotama pravděpodobnosti pro daný žánr. Vybere se 5 nejvíce hodících a z těch se následně vybírá nejpodobnější náladě a tempu.