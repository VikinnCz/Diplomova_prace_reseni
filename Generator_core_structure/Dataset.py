class Dataset:

    def __init__(self, genre : dict, tempo : int, mood : int, anim_blocks):
        self._genre = genre
        self._tempo = tempo
        self._mood = mood
        self._anim_blocks = anim_blocks

    @property
    def genre(self):
        return self._genre
    
    @property
    def tempo(self):
        return self.__tempo
    
    @property
    def mood(self):
        return self.__mood

    @property
    def anim_blocks(self):
        return self.__anim_blocks