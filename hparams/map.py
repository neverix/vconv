"""
Map data structure with dot access.
"""


class Map(dict):
    """
    Map data structure with dot access.
    Adapted from https://gist.github.com/miku/dc6d06ed894bc23dfd5a364b7def5ed8#file-23689767-py
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Map(v)
