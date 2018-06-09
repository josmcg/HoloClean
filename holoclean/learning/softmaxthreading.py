from threading import Thread, Lock

printLock = Lock()


def safe_print(msg):
    printLock.acquire()
    print msg
    printLock.release()
    return


class ModuleThread(Thread):

    __lock = Lock()

    def __init__(self, featurizer):
        Thread.__init__(self)
        self.tensor = None
        self.featurizer = featurizer
        return

    def run(self):
        self.tensor = self.featurizer.forward()
        return
