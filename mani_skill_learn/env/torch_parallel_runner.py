"""
Multiprocess that supports pytorch, which will consume a lot of memory due to cuda and pytorch.
"""
from torch.multiprocessing import Process
from .parallel_runner import WorkerBase


class TorchWorker(WorkerBase, Process):
    def __init__(self, *args, **kwargs):
        Process.__init__(self)
        WorkerBase.__init__(self, True, *args, **kwargs)
