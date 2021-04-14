import time


class Timer():
    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc = time.time()
        print(self.toc-self.tic)
