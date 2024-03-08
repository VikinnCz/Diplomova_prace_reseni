import librosa
import madmom
import numpy as np
import scipy
import colorsys

CHILL       = 0
HANG_OUT    = 1
HAPPY       = 2
DANCING     = 3

class ChromaFeatures:
    def __init__(self, file_name, mood):
        self.__mood = mood
        self.__file_name = file_name
        self.__Calc_color_palette()

    def __Calc_chroma_librosa(self):
        y, sr = librosa.load(self.__file_name)

        y_harm = librosa.effects.harmonic(y=y, margin=8)

        chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
        chroma_filter = np.minimum(chroma_harm, librosa.decompose.nn_filter(chroma_harm, aggregate=np.median, metric= 'cosine'))
        self.__chroma = scipy.ndimage.median_filter(chroma_filter, size=(1,9))
    
    def __Calc_chroma_madmom(self):
        dcp = madmom.audio.DeepChromaProcessor()
        chroma_madmom_deep = dcp(self.__file_name)
        self.__chroma = np.transpose(chroma_madmom_deep)

    def __Calc_color_palette(self):
        match self.__mood:
            case 0:
                self.__Calc_chroma_madmom()
                chroma_binary = (self.__chroma > 0.6).astype(int)
            case 1:
                self.__Calc_chroma_madmom()
                chroma_binary = (self.__chroma > 0.6).astype(int)
            case 2: 
                self.__Calc_chroma_librosa()
                chroma_binary = (self.__chroma > 0.6).astype(int)
            case 3:
                self.__Calc_chroma_librosa()
                chroma_binary = (self.__chroma > 0.6).astype(int)

        tones = np.sum(chroma_binary, axis=1)

        tones_copy = np.copy(tones)
        most_played_tones = [0,0,0]

        for i in range(3):
            most_played_tones[i] = np.argmax(tones_copy)
            tones_copy[most_played_tones[i]] = 0


        hue = np.random.randint(0, 360)  # Náhodný odstín
        triadic_colors = self.__Generate_triadic_colors(hue)
        self.__tones_colors = np.zeros(shape=(12,3), dtype=int)

        for i in range(3):
            self.__tones_colors[most_played_tones[i]] = triadic_colors[i]

        for i in range (12):
            for n in range (3):
                try:
                    if np.sum(self.__tones_colors[most_played_tones[n]-i]) == 0 and (most_played_tones[n]-i) >= 0:
                        self.__tones_colors[most_played_tones[n]-i] = self.__Color_shift(triadic_colors[n],i);
                        # self.__tones_colors[most_played_tones[n]-i] = np.rint(triadic_colors[n] * (1-(i*0.08))).astype(int)
                except IndexError:
                    continue

                try:
                    if np.sum(self.__tones_colors[most_played_tones[n]+i]) == 0:
                        self.__tones_colors[most_played_tones[n]+i] = self.__Color_shift(triadic_colors[n],i);
                        # self.__tones_colors[most_played_tones[n]+i] =np.rint(triadic_colors[n] * (1-(i*0.08))).astype(int)
                except IndexError:
                    continue

    def __Generate_triadic_colors(self, hue, saturation=0.75, lightness=0.5):
        """Generuje triadické barvy zadané HSL hodnoty."""
        # Správná syntaxe pro list comprehension
        colors_hsl = [((hue + i * 120) % 360, saturation, lightness) for i in range(3)]
        colors_rgb = [colorsys.hls_to_rgb(h/360.0, lightness, saturation) for h, s, l in colors_hsl]
        # Převedení RGB z rozsahu 0-1 na 0-255
        colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors_rgb]
        colors_rgb = np.array(colors_rgb)
        return colors_rgb
    
    def __Color_shift(self, old_color, i):
        old_color_hsv = colorsys.rgb_to_hsv(r=old_color[0],g=old_color[1],b=old_color[2])
        new_color_hsv = list(old_color_hsv)
        new_color_hsv[0] = (new_color_hsv[0] + (i * 15)/360.0)
        new_color = colorsys.hsv_to_rgb(new_color_hsv[0],new_color_hsv[1], new_color_hsv[2])
        new_color = [int(new_color[0]), int(new_color[1]), int(new_color[2])]
        return new_color
    
    def Get_tones_colors(self):
        return self.__tones_colors
    
    def Get_chroma(self):
        return self.__chroma