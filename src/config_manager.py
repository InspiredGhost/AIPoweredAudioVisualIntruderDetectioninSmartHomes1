# Minimal config manager placeholder
class Config:
    def __init__(self):
        self.settings = {}

    def get(self, key, default=None):
        return self.settings.get(key, default)

config = Config()
