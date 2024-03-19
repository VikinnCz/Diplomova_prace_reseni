import librosa
import madmom
import numpy as np
import scipy
import colorsys
import Constants

class ChromaFeatures:
    """ A class for chroma freature analysis.

    This class compute a chromagram from musinc song and provides an overview of its chroma features. Two different approaches are used for the analysis.
    First approach is from library Librosa that compute chromagram based on CQT chroma analysis. Second approach is from library Madom that use deep neural network that focuses on harmonically relevant spectral content.
    
    The models for computing chromagram are selected based on mood constants.

    Atributes
    ----------
    audio_path: str
        Address for localization of audio file to analyze.
    mood : int
        Mood of the music song.
    chroma : array
        Computed chromagram.
    tones_colors : array
        Computed color palete based on the crhom features.

    Methods
    ----------
    Get_tones_colors(self)
        Return array of color palete based on chroma features of music song.
    Get_chroma(self)
        Return chromagram in array values.
        
    """
    def __init__(self, audio_path, mood):
        """
        Parameters
        ----------
        audio_path : str
            Address for lacalization of audio file to analyze.
        mood : int
            Mood of the music song.
        """
        self.__mood = mood
        self.__audio_path = audio_path
        self.__Calc_color_palette()

    def __Calc_chroma_librosa(self):
        """
        Calculates chromagram based on Librosa library.

        Librosa library compute chromagram based on CQT chroma analysis. The function use preprocesing methods a postprocesing methods to improve analysis results.
        """
        y, self.__sr = librosa.load(self.__audio_path)

        y_harm = librosa.effects.harmonic(y=y, margin=8) # Preprocesing by extraction haromic elements from audio.

        chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=self.sr) # CQT chromagram calculation.
        chroma_filter = np.minimum(chroma_harm, librosa.decompose.nn_filter(chroma_harm, aggregate=np.median, metric= 'cosine'))
        self.__chroma = scipy.ndimage.median_filter(chroma_filter, size=(1,9)) # Chromagram filterign thru chroma_filter.
    
    def __Calc_chroma_madmom(self):
        """
        Calculates chromagram based on Madmom library.

        Madmom library compute chromagram using a deep neural network that focuses on harmonically relevant spectral content.
        """
        dcp = madmom.audio.DeepChromaProcessor()
        chroma_madmom_deep = dcp(self.__audio_path,fps = 20.751)
        self.__chroma = np.transpose(chroma_madmom_deep)

    def __Calc_color_palette(self):
        """
        Calculation of color palete from chromagram.

        This method calculated color palete based on most sounded tones in audio. To the first three tones are assigned random colors with hue spacing of 120 degree. To others tones are asigned colors with hue spacing 10 degree from the base color. 
        """

        # Caall method for chromagram calculation based on the mood.
        match self.__mood:
            case Constants.CHILL:
                self.__Calc_chroma_madmom()
            case Constants.HANG_OUT:
                self.__Calc_chroma_madmom()
            case Constants.HAPPY: 
                self.__Calc_chroma_librosa()
            case Constants.DANCING:
                self.__Calc_chroma_librosa()

        chroma_binary = (self.__chroma > 0.6).astype(int) # Transform chromagram from float to binary values.

        tones = np.sum(chroma_binary, axis=1) # Compute how much the tones are sounded.

        tones_copy = np.copy(tones)
        most_played_tones = [0,0,0]

        # Take the three most sounded tones.
        for i in range(3):
            most_played_tones[i] = np.argmax(tones_copy)
            tones_copy[most_played_tones[i]] = 0


        hue = np.random.randint(0, 360)  # Generate random hue of collor.
        triadic_colors = self.__Generate_triadic_colors(hue)
        self.__tones_colors = np.zeros(shape=(12,3), dtype=int)

        # Add to the color palete the most playd tones colors.
        for i in range(3):
            self.__tones_colors[most_played_tones[i]] = triadic_colors[i]

        # Add to the color palete shades of other tones colors.
        for i in range (12):
            for n in range (3):
                try:
                    if np.sum(self.__tones_colors[most_played_tones[n]-i]) == 0 and (most_played_tones[n]-i) >= 0:
                        self.__tones_colors[most_played_tones[n]-i] = self.__Color_shift(triadic_colors[n],i)
                except IndexError:
                    continue

                try:
                    if np.sum(self.__tones_colors[most_played_tones[n]+i]) == 0:
                        self.__tones_colors[most_played_tones[n]+i] = self.__Color_shift(triadic_colors[n],i)
                except IndexError:
                    continue

    def __Generate_triadic_colors(self, hue, saturation=0.85, lightness=0.5):
        """
        Generate triadic colors based on hue of first color.

        This function generates three colors with hue spacing of 120 degree. Know as triadic colors. 

        Parameters
        ----------
        hue : int

        saturation : float, optional

        lightness : float, optional

        Returns
        ----------
        colors_rgb : NDArray[int]
            Array of three rgb colors.
        """
        # Colors generating
        colors_hsl = [((hue + i * 120) % 360, saturation, lightness) for i in range(3)]
        colors_rgb = [colorsys.hls_to_rgb(h/360.0, lightness, saturation) for h, s, l in colors_hsl]

        # Colors value transform from range 0-1 to range 0-255
        colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors_rgb]
        colors_rgb = np.array(colors_rgb) # Transform from tuple to array

        return colors_rgb
    
    def __Color_shift(self, old_color, i):
        """
        Color shifting

        Function shifts hue of the old color by 15 * i degree 

        Parameters
        ----------
        old_color : array[int]
            RGB values of color to shift.
        i : int
            An index indicating how much the color should be shifted.

        Returns
        ----------
        array[int]
            RGB values of shifted color.
        """
        old_color_hsv = colorsys.rgb_to_hsv(r=old_color[0],g=old_color[1],b=old_color[2])
        new_color_hsv = list(old_color_hsv)
        new_color_hsv[0] = (new_color_hsv[0] + (i * 15)/360.0) # Hue shift 
        new_color = colorsys.hsv_to_rgb(new_color_hsv[0],new_color_hsv[1], new_color_hsv[2])
        new_color = [int(new_color[0]), int(new_color[1]), int(new_color[2])]
        return new_color
    

    @property
    def sr(self):
        return self.__sr
    
    @property
    def tones_colors(self):
        return self.__tones_colors
    
    @property
    def crhoma(self):
        return self.__chroma