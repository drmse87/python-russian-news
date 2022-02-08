class Document:
    def __init__(self, label, contents):
        self._label = label
        self._contents = contents

    @property
    def label(self):
        return self._label

    @property
    def contents(self):
        return self._contents