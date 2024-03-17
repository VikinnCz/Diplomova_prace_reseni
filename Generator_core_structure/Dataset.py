class Dataset:

    def __init__(self, genre : int, speed_suitability : int, mood_characteristics : int, animation_blocks = 0):
        self._genre = genre
        self._speed_suitability = speed_suitability
        self._mood_characteristics = mood_characteristics
        self._animation_blocks = animation_blocks

    @property
    def genre(self):
        return self._genre
    
    @property
    def speed_suitability(self):
        return self.__speed_suitability
    
    @property
    def mood_characteristics(self):
        return self.__mood_characteristics

    @property
    def animation_blocks(self):
        return self.__animation_blocks