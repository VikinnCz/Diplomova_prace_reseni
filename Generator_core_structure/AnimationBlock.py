class AnimationBlock:
    def __init__(self, anim_code : str, anim_characteristics : int, anim_length : int, anim_colors : int):
        self.__code = anim_code
        self.__characteristics = anim_characteristics
        self.__length = anim_length
        self.__colors = anim_colors

    @property
    def anim_code(self):
        return self.__code
    
    @property
    def anim_characteristics(self):
        return self.__characteristics
    
    @property
    def anim_length(self):
        return self.__length
    
    @property
    def anim_color(self):
        return self.__colors
    
    def to_dict(self):
        return{
            "anim_code" : self.__code,
            "anim_characteristics" : self.__characteristics,
            "anim_length" : self.__length,
            "anim_colors" : self.__colors}
