# test_low_level_threads.py
import time
from multiprocessing import TimeoutError as MpTimeoutError
from queue import Empty as Queue_Empty
from queue import Queue
from _thread import start_new_thread
from ctypes import c_long
from ctypes import py_object
from ctypes import pythonapi


class MyTimeoutError(Exception):
    pass


def async_raise(tid, exctype=Exception):
    """
    Raise an Exception in the Thread with id `tid`. Perform cleanup if
    needed.
    Based on Killable Threads By Tomer Filiba
    from http://tomerfiliba.com/recipes/Thread2/
    license: public domain.
    """
    assert isinstance(tid, int), "Invalid  thread id: must an integer"

    tid = c_long(tid)
    exception = py_object(Exception)
    res = pythonapi.PyThreadState_SetAsyncExc(tid, exception)
    if res == 0:
        raise ValueError("Invalid thread id.")
    elif res != 1:
        # if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect
        pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed.")


def timeout_func(func, args=None, kwargs=None, timeout=30, q=None):
    """
    Threads-based interruptible runner, but is not reliable and works
    only if everything is pickable.
    """
    # We run `func` in a thread and block on a queue until timeout
    if not q:
        q = Queue()

    def runner():
        try:
            _res = func(*(args or ()), **(kwargs or {}))
            q.put((None, _res))
        except MyTimeoutError:
            # rasied by async_rasie to kill the orphan threads
            pass
        except Exception as ex:
            q.put((ex, None))

    tid = start_new_thread(runner, ())

    try:
        err, res = q.get(timeout=timeout)
        if err:
            raise err
        return res
    except (Queue_Empty, MpTimeoutError):
        return args[0]
    finally:
        try:
            async_raise(tid, MyTimeoutError)
        except (SystemExit, ValueError):
            pass


def _fit_distribution(length=50):
    # some long operation which run for undetermined time
    print(f"length is {length}")
    for i in range(length):
        time.sleep(i)

    print("woke up")
    return "succes"

if __name__ == "__main__":
    try:
        for _ in range(50):
            res = timeout_func(_fit_distribution, args=(50,), timeout=0.03)
            print(f"we got {res}")
    except MyTimeoutError as ex:
        print(ex)
