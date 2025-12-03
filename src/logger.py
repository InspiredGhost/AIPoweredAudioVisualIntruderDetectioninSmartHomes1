# Minimal logger placeholder

def get_logger(name=None):
    class Logger:
        def info(self, msg):
            print(f"INFO: {msg}")
        def warning(self, msg):
            print(f"WARNING: {msg}")
        def error(self, msg):
            print(f"ERROR: {msg}")
    return Logger()
