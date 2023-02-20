
import os
from pathlib import Path

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if not Path(self.filepath).parent.exists():
            os.makedirs(Path(self.filepath).parent)
        if Path(self.filepath).exists():
            os.remove(self.filepath)
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                print(str)
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()
