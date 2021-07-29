from copy import deepcopy
from inspect import isfunction
from multiprocessing import Process, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class WorkerBase:
    ASK = 1
    CALL = 2
    GETATTR = 3
    GET = 4
    EXIT = 5

    def __init__(self, use_torch, cls, worker_id, *args, **kwargs):
        self.cls = cls
        self.worker_id = worker_id
        self.args = deepcopy(args)
        self.kwargs = deepcopy(dict(kwargs))
        self.kwargs['worker_id'] = worker_id
        if use_torch:
            from torch.multiprocessing import Pipe
        else:
            from multiprocessing import Pipe
        self.pipe, self.worker_pipe = Pipe()
        self.daemon = True

        if hasattr(self, 'start'):
            self.start()
        else:
            print('We should merge this class to another class')
            exit(0)

    def run(self):
        is_object = False
        if isfunction(self.cls):
            func = self.cls
        else:
            is_object = True
            func = self.cls(*self.args, **self.kwargs)
        ans = None
        while True:
            op, args, kwargs = self.worker_pipe.recv()
            if op == self.ASK:
                ans = func(*args, **kwargs)
            elif op == self.CALL:
                assert is_object
                func_name = args[0]
                args = args[1]
                ans = getattr(func, func_name)(*args, **kwargs)
            elif op == self.GETATTR:
                assert is_object
                ans = getattr(func, args)
            elif op == self.GET:
                self.worker_pipe.send(ans)
            elif op == self.EXIT:
                if func is not None and is_object:
                    del func
                self.worker_pipe.close()
                return

    def call(self, func_name, *args, **kwargs):
        self.pipe.send([self.CALL, [func_name, args], kwargs])

    def get_attr(self, attr_name):
        self.pipe.send([self.GETATTR, attr_name, None])

    def ask(self, *args, **kwargs):
        self.pipe.send([self.ASK, args, kwargs])

    def get(self):
        self.pipe.send([self.GET, None, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([self.EXIT, None, None])
        self.pipe.close()


class NormalWorker(WorkerBase, Process):
    def __init__(self, *args, **kwargs):
        Process.__init__(self)
        WorkerBase.__init__(self, False, *args, **kwargs)


def split_list_of_parameters(num_procsess, *args, **kwargs):
    from mani_skill_learn.utils.math import split_num
    args = [_ for _ in args if _ is not None]
    kwargs = {_: __ for _, __ in kwargs.items() if __ is not None}
    assert len(args) > 0 or len(kwargs) > 0
    first_item = args[0] if len(args) > 0 else kwargs[list(kwargs.keys())[0]]
    n, running_steps = split_num(len(first_item), num_procsess)
    start_idx = 0
    paras = []
    for i in range(n):
        slice_i = slice(start_idx, start_idx + running_steps[i])
        start_idx += running_steps[i]
        args_i = list([_[slice_i] for _ in args])
        kwargs_i = {_: kwargs[_][slice_i] for _ in kwargs}
        paras.append([args_i, kwargs_i])
    return paras
