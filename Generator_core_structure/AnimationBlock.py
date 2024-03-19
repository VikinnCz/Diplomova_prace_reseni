class AnimationBlock:
    def __init__(self, anim_code : str, anim_characteristics : int, anim_length : int):
        self._anim_code = anim_code
        self._anim_characteristics = anim_characteristics
        self._anim_length = anim_length

    @property
    def anim_code(self):
        return self._anim_code
    
    @property
    def anim_characteristics(self):
        return self._anim_characteristics
    
    @property
    def anim_length(self):
        return self._anim_length
