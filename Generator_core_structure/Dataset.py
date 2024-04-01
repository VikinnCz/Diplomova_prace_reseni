class Dataset:

    def __init__(self, genre : dict, tempo : int, mood : int, anim_blocks):
        self.__genre = genre
        self.__tempo = tempo
        self.__mood = mood
        self.__anim_blocks = anim_blocks

    @property
    def genre(self):
        return self.__genre
    
    @property
    def tempo(self):
        return self.__tempo
    
    @property
    def mood(self):
        return self.__mood

    @property
    def anim_blocks(self):
        return self.__anim_blocks
    
    def to_dict(self):
        return {
        "genre": self.__genre,
        "tempo": self.__tempo,
        "mood": self.__mood,
        "anim_blocks": self.__anim_blocks}