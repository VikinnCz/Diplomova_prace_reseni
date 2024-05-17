import os
import json
import Constants as cns
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from io import BytesIO

from Dataset import Dataset
from Segmentation import Segmentation
from BeatTracking import BeatTracking
from AnimationBlock import AnimationBlock
from ChromaFeatures import ChromaFeatures
from GenreClassification import GenreClassification

from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'FFFFFFFFFFFFFFFF'
app.config["DEBUG"] = True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'submit_button' in request.form:
        # Zkontrolujte, zda soubor byl odeslán
        if 'audiofile' in request.files and request.files['audiofile'].filename != '':
            # try:
                file = request.files['audiofile']
                slider_value = request.form['slider']

                # Processing
                file_stream = BytesIO(file.read())
                y, sr = sf.read(file_stream, dtype="float32")

                spectoda_code = main(y=y, sr=sr, slider_value=slider_value)

                # Send generated code to html template
                session['processed_data'] = spectoda_code
                return redirect(url_for('index'))
            # except:
            #     session['processed_data'] = "Nastala chyba při generování animace"
            #     return redirect(url_for('index'))
        else:
            session['processed_data'] = "Nebyl vybrán žádný audio soubor."
            return redirect(url_for('index'))

    processed_data = session.pop('processed_data', None)
    if processed_data != None:
        return render_template('main_page.html', code=processed_data)

    return render_template('main_page.html', code="Zde se po nahrání souboru a stiknutí talačítka generovat zobrazí vygenerovaný Spectoda kód animace. Proces analýzy a generování skladby může chvíli trvat.")


def main(y, sr, slider_value):

    dataset_database = Load_dataset_database()
    mood = int(slider_value)

    if is_stereo(y=y):
        y =  np.mean(y, axis=1)

    if sr != 22050:
        y = librosa.resample(y=y,orig_sr=sr, target_sr=22050,)
        sr = 22050

    beat_tracking = BeatTracking(y=y, sr=sr)
    chroma_features = ChromaFeatures(y=y, sr=sr, mood=mood)
    genre_classification = GenreClassification(y=y, sr=sr)
    segmentation = Segmentation(chroma_features=chroma_features, mood=mood)

    dataset = Dataset_selection(dataset_database, genre_classification, beat_tracking, mood)

    spectoda_code = Code_generation(y=y,
                                    sr=sr,
                                    dataset=dataset,
                                    beat_tracking=beat_tracking,
                                    chroma_features=chroma_features,
                                    segmentation=segmentation)
    return spectoda_code

def is_stereo(y):
    if y.ndim == 1:
        return False  # Audio je mono
    elif y.ndim == 2 and y.shape[1] == 1:
        return False  # Audio je také mono, ale s explicitním jedním kanálem
    else:
        return True  # Audio má více kanálů (stereo nebo vícekanálové)

def Code_generation(y, sr, dataset:Dataset, beat_tracking:BeatTracking, chroma_features:ChromaFeatures, segmentation:Segmentation):
    """
    Main function that generates spectoda code of animation. The function run through all beats and segments of audio a assignins anim_blocks to this beats. Function uses others analyzed parameters to calculate time, colors and suitability of the animation block.

    Parameters
    ----------
    y : array
        Samples of audio for analyze.
    sr : float
        Sample rate of audio for analyze.
        beat_tracking
    beat_tracking : BeatTracking
        Object of analyzed beats
    chroma_features : ChromaFeatures
        Object of analyzed chroma features
    segmentation : Segmentation
        Object of calculated segments

    """

    os.system("cls")

    beats = beat_tracking.beats
    beats_strength = beat_tracking.strength
    times = beat_tracking.times
    chroma = chroma_features.crhoma
    tones_colors = chroma_features.tones_colors
    segments = segmentation.bounds # [s]

    beats_stength_median = Calc_median(beats_strength)
    audio_loudness = Calc_loudness(y, sr)
    audio_duration = len(y) / sr # [s]
    print(f"Audio duration : {audio_duration}")

    timeline_animations = []
    completed = False

    start_beat = beats[0]
    start_beat_index = 0
    while not completed:
    
        # V jakém segmentu se nachází start_beat
        start_segment, end_segment = Find_segment(segments, start_beat)

        if end_segment == None:
            end_segment = audio_duration

        segment_duration = end_segment - start_beat # [s]
        print(f"Segment duration : {segment_duration}")

        # Hlasitost segmentu
        segment_loudness = Calc_loudness(y=y, sr=sr, start_time=start_segment, end_time=end_segment)

        # Beat parametry
        # Porovnání síly beatu s mediánem sil beatů ve skladbě.
        start_beat_strength = beats_strength[start_beat_index]


        # Porovnání síly beatu s mediánem sil beatů ve sklabě.
        segment_end_beat_index = Find_nearest_beat(end_segment, beats)
        segment_beat_strength_median = Calc_median(y=beats_strength,
                                                    start_beat_index=start_beat_index,
                                                    end_beat_index=segment_end_beat_index)

        # Čas k dalšímu nejpodobnějšímu beatu v segmentu.
        # next_similar_beat =  beats[Find_next_similar_beat(beats_strength, start_beat_index, segment_end_beat_index)]
        # time_to_similar_beat = next_similar_beat - start_beat
        # print(f"Time to similar beat : {time_to_similar_beat}")

        # Logika pro vybrání anim bloku ze zjištěných parametrů
        is_long = None
        is_loud = None
        is_important_in_audio = None
        is_important_in_segment = None

        # Segment
        if (segment_duration >= 0.04*audio_duration):
            # Dlouhý segment
            is_long = True
        else:
            is_long = False
            # Krátký segment

        if(segment_loudness >= audio_loudness/0.7):
            # Hlasitý segment
            is_loud = True
        else:
            # Tichý segment
            is_loud = False
        # Beat
        if(start_beat_strength >= 0.5*beats_stength_median):
            # Významný v nahrávce
            is_important_in_audio = True
        else:
            # Nevýznamný v nahrávce
            is_important_in_audio = False

        if(start_beat_strength >= 0.7*segment_beat_strength_median):
            # Významný v segmentu
            is_important_in_segment = True
        else:
            # Nevýznamný v segmentu
            is_important_in_segment = False

        anim_char = Char_selection(
                    is_long = is_long,
                    is_loud = is_loud,
                    is_important_in_audion = is_important_in_audio,
                    is_important_in_segment = is_important_in_segment)
        # print(f"Anim char is : {anim_char}")
        anim_blocks = dataset.anim_blocks
        anim_block = next(block for block in anim_blocks if block.anim_characteristics == anim_char)
        
        # Délka animace
        anim_duration = anim_block.anim_length
        end_beat = beats[Find_nearest_beat(y=beats, time=start_beat+anim_duration)]
        
        # Modifikace času animace
        time_to_end_beat = end_beat - start_beat
        if time_to_end_beat == 0:
            completed = True
            time_to_end_beat = audio_duration - start_beat

        time_modifier = anim_duration/time_to_end_beat
        # print(f"Time modifier : {time_modifier}")

        # Přiřazení barev
        color_range = anim_block.anim_color
        anim_code = anim_block.anim_code
        primary_tone = []
        colors = []

        time_index_start =np.where(times == start_beat)[0]
        time_index_end = np.where(times == end_beat)[0]

        chroma_in_range = np.sum(chroma[:,time_index_start[0]:time_index_end[0]], axis=1)

        # n barev dle color_range bloku animace
        for i in range(color_range):
            primary_tone.append(chroma_in_range.argmax())
            chroma_in_range[primary_tone[i]] = 0

        for i in range(len(primary_tone)):
            index = int(primary_tone[i])
            tone_color = tones_colors[index]
            colors.append('#%02x%02x%02x' % (tone_color[0], tone_color[1], tone_color[2]))

        # Vygenerování spectoda codu animace.
        match color_range:
            case 0:
                anim_code = anim_code.format(start_beat = start_beat,
                            end_beat = end_beat,
                            time_modifier = time_modifier)
            case 1:
                anim_code = anim_code.format(start_beat = start_beat,
                            end_beat = end_beat,
                            time_modifier = time_modifier,
                            color_0 = colors[0])
            case 2:
                anim_code = anim_code.format(start_beat = start_beat,
                            end_beat = end_beat,
                            time_modifier = time_modifier,
                            color_0 = colors[0],
                            color_1 = colors[1])
            case 3:
                anim_code = anim_code.format(start_beat = start_beat,
                            end_beat = end_beat,
                            time_modifier = time_modifier,
                            color_0 = colors[0],
                            color_1 = colors[1],
                            color_2 = colors[2])    
        timeline_animations.append(anim_code)

        # Nastavení následujícího start_beat jako momentální end_beat
        start_beat = end_beat
        start_beat_index = Find_nearest_beat(beats, end_beat)
        print(f"End beat time: {end_beat}")

        # Kontrolní podmínka jestli už je vygenerovaná animace pro celou skladbu'
        if end_beat > audio_duration or completed:
            completed = True
            print("Anim generation COMPLETD")




    # Staré řešení pro umísťování animací na beaty.
    #_________________________________________________
    # for index, beat in enumerate(beats):

    #     if beats_strength[index] > (beats_stength_median-0.2):
    #         range = 50
            
    #         time_index =np.where(times == beat)[0]
    #         if time_index - range >= 0 and time_index + range < chroma.shape[1]:
    #             chroma_in_range = np.sum(chroma[:,int(time_index-range):int(time_index+range)], axis=1)
    #             primary_tone = chroma_in_range.argmax()
    #         else:
    #             primary_tone = chroma[:,time_index].argmax() # Tohle je funkcni bez casoveho rozmezi
             

    #         tone_color = tones_colors[primary_tone]
    #         hex_tone_color = '#%02x%02x%02x' % (tone_color[0], tone_color[1], tone_color[2])

    #         timeline_animations.append(f"addDrawing({beat:.2f}s, 0.5s, animPlasmaShot(0.5s, {hex_tone_color}, 25%));")
    #___________________________________________________

    print (f"Počet vygenerovaných animací: {len(timeline_animations)}")
    return ''.join(timeline_animations)

def Find_segment(y : list, beat_time):
    """
    Function that finds segment from provided list in which is located the beat_time.

    Parameters
    ----------
    y : list
        List of times where the segments boundaries are located
    beat_time : float
        Time of the beat which is wanted to locate.

    Returns
    ----------
    start_segment : float
        Time where the located segment starts.
    end_segment : float
        Time where the located segment ends.
    """
    y = np.asarray(y)
    idx = np.abs((y - beat_time)).argmin()

    if y[idx] > beat_time:
        idx -= 1

    start_segment = y[idx]
    try:
        end_segment = y[idx+1]
    except IndexError:
        end_segment = None

    return start_segment, end_segment

def Find_nearest_beat(y : list, time):
    """
    This function finds the nearest beat in given list.

    Parameters
    ----------
    y : list
        List of times where the beats are located.
    time : float
        Time around that is searching for the nearest beat.

    Returns 
    ----------
    idx : int
        Index of finded beat in the list.
    """
    y = np.asarray(y)
    idx = np.abs((y - time)).argmin() 
    return idx

def Find_next_similar_beat(y : list, beat_index : int, end_beat : int):
    """
    Function that find the nears similar beat to given beat.

    Parameters
    ----------
    y : list
        List of times where the beats are located.
    beat_index : int
        Index in list of the given beat.
    end_beat : int
        Index of the beat to which is searched.

     Returns 
    ----------
    idx : int
        Index of finded most simillar beat in the list.
    beat_index : int 
        Index of given beat which is returned if in the list are no mor beats.
    """
    start_beat_strength = y[beat_index]
    difference = 255

    try:
        for index, beat_strength in enumerate(y[beat_index+1 : end_beat]):
            n = np.abs(start_beat_strength - beat_strength)
            if (difference > n):
                difference = n
                idx = index
        return idx
    except UnboundLocalError:
        return beat_index

def Calc_loudness(y, sr, start_time = None, end_time = None):
    """
    Function which calculate loudness in LUFS for given date. Function can calculate for all the data or in given segment.

    Parameters
    ----------
    y : list
        Audio samples
    sr : int
        Sampling rate of audio samples
    start_time : float | None
        Start time of the segment in which are the loudness calculated.
    end_time : float | None
        End time of the segment in which are the loudness calculated.

    Returns 
    ----------
    loudness : float
        Calculated loudness
    """
    if(start_time == None or end_time == None):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        return loudness
    else:
        start_sample = int(sr*start_time)
        end_sample = int(sr*end_time)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y[start_sample:end_sample])
        return loudness

def Calc_median(y, start_beat_index = None, end_beat_index = None):
    if(start_beat_index == None or end_beat_index == None):
        median = np.median(y)
        return median
    else:
        median = np.median(y[start_beat_index:end_beat_index])
        return median
    
def Char_selection(is_long, is_loud, is_important_in_audion, is_important_in_segment):
    match [is_long, is_loud]:
        case[False, True]:
            #Krátké úderné
            match[is_important_in_audion,is_important_in_segment]:
                case[False, True]:
                    return cns.SHOT
                case[False, False]:
                    return cns.PULL
                case[True, True]:
                    return cns.BANG
                case[True, False]:
                    return cns.SHOT
                    
        case[False, False]:
            #Krátké pohodové
            match[is_important_in_audion,is_important_in_segment]:
                case[False, True]:
                    return cns.PULL
                case[False, False]:
                    return cns.THEMA
                case[True, True]:
                    return cns.SHOT
                case[True, False]:
                    return cns.PULL
        case[True, True]:
            #Dlouhé úderné
            #TODO: Možné dodělat pokud je start beat na začátku segmentu je automatikc považován za PULL.
            match[is_important_in_audion,is_important_in_segment]:
                case[False, True]:
                    return cns.PULL
                case[False, False]:
                    return cns.THEMA
                case[True, True]:
                    return cns.SHOT
                case[True, False]:
                    return cns.PULL
        case[True, False]:
            #TODO: Možné dodělat pokud je start beat na začátku segmentu je automatikc považován za THEMA.
            #Dlouhé táhnoucí se
            match[is_important_in_audion,is_important_in_segment]:
                case[False, True]:
                    return cns.FLOW
                case[False, False]:
                    return cns.FLOW
                case[True, True]:
                    return cns.PULL
                case[True, False]:
                    return cns.THEMA

def Dataset_selection(dataset_database : list[Dataset], genre_classification : GenreClassification, beat_tracking : BeatTracking, mood : int):
    """
    Function that return dataset based on audio genre, tempo, and user mood. Function search thru all datasets and compare values of genre prediction. First five datasets with smallest differences in genre prediction values are selected. Then are selected datasets with the same mood and nearest tempo value. 

    Parameters
    ----------
    dataset_database : list[Dataset]
        Database of datasets. 
    genre_classification : GenreClassification
        Object with genre predictions data from the audio
    beat_tracking : BeatTracking
        Object with Beat tracking data from the audio
    mood : int
        User selected mood.
    
    Returns 
    ----------
    selected_dataset : Dataset
        Dataset which is selected as most suitable.
    """
    
    # Get parameters 
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

    # Get five datasets with smallest genre difference
    for i in range(5):
        index_of_min = int(np.argmin(genres_difs))
        genre_pass_datasets.append( dataset_database[index_of_min])
        genres_difs[index_of_min] = 255

    this_tempo_dif = 100
    selected_dataset = Dataset
    
    # Get dataset with same mood an smallest tempo diference
    for dataset in genre_pass_datasets:
        if dataset.mood == mood:
            new_tempo_dif = abs(dataset.tempo - tempo)
            if this_tempo_dif > new_tempo_dif:
                this_tempo_dif = new_tempo_dif
                selected_dataset = dataset

    return selected_dataset

def Load_dataset_database():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open (os.path.join(__location__, "dataset_database.json"), "r") as fp:
        dataset_database_json = json.load(fp,)

    dataset_database = [Dataset(**dataset) for dataset in dataset_database_json]
    return dataset_database


if __name__ == "__main__":

    app.run(debug=True)
    # audio_path = "Referencni_skladby/Imanbek & BYOR - Belly Dancer (Official Music Video).wav"
    # audio_path = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/07.-Here Comes The Sun.wav"
    # audio_path = "Referencni_skladby/The Beatles - Abbey Road (1969) (2012 180g Vinyl 24bit-96kHz) [FLAC] vtwin88cube/01.-Come Together.wav"
    # audio_path = "Referencni_skladby/BABYMONSTER - SHEESH MV.mp3"
    # audio_path = "Referencni_skladby/Benson Boone - Beautiful Things (Official Music Video).mp3"

    # dataset_database = Load_dataset_database()
    # mood = cns.HAPPY

    # y, sr = librosa.load(path=audio_path, sr=22050, mono=True)

    # beat_tracking = BeatTracking(y=y, sr=sr)
    # chroma_features = ChromaFeatures(y=y, sr=sr, mood=mood)
    # genre_classification = GenreClassification(y=y, sr=sr)
    # segmentation = Segmentation(chroma_features=chroma_features)
    # segmentation.bounds
    

    # dataset = Dataset_selection(dataset_database, genre_classification, beat_tracking, mood)

    # timeline_animations = Code_generation(y, sr, beat_tracking, chroma_features, segmentation)
    # spectoda_code = ''.join(timeline_animations)
    # print(spectoda_code)


## Parametry které je možné nastavovat a na základě toho měnit generování animací. ##
# Na základě mood můžu nastavovat trashold pro beat strenght
# Počet generovaných segmentů (number_of_segments)


#Jak vybírat datasety abych pro každý žánr nemusel mít dataset pro každou náladu (40 datasetů). 
    # Řešení 1: žánr bude mít nejnižší prioritu při výběru. Prioritní bude mood a tempo až nakonec žánr.
    # Řešení 2: parametr genre bude list a každý dataset může být pro několik žánrů. Například to může být pole s hodnotou pro každý žánr a tato hodnota bude určovat vhodnost pro daný žánr. Tato hodnota se pak porovnává s hodnotama pravděpodobnosti pro daný žánr. Vybere se 5 nejvíce hodících a z těch se následně vybírá nejpodobnější náladě a tempu.