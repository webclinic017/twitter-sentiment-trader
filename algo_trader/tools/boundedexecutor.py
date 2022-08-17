# internal
import operator

# external
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore
from time import sleep

from .errors import ThreaderFailed, err_catch
from .separators import dashes


class BoundedThreads:
    """BoundedThreads behaves as a ThreadPoolExecutor which will block on
    calls to submit() once the limit given as "bound" work items are queued for
    execution.
    :param bound: Integer - the maximum number of items in the work queue
    :param max_workers: Integer - the size of the thread pool
    """

    def __init__(self, bound=4, max_workers=100, thread_name_prefix="", buffer=0.1):
        """
        The __init__ function is called when an instance of the class is created.
        It initializes variables that are specific to each instance, and sets up a logger.

        :param self: Refer to the instance of the class
        :param bound=4: Set the maximum number of threads that can be running at any given time
        :param max_workers=100: Set the maximum number of threads that can be used at any given time
        :param thread_name_prefix='': Set the prefix for the thread names
        :return: The object that was created
        :doc-author: Trelent
        """
        self.tag = thread_name_prefix
        self.buffer = buffer
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )
        self.semaphore = BoundedSemaphore(bound + max_workers)
        self.futures = dict()

    """See concurrent.futures.Executor#submit"""

    def submit(self, fn, *args, **kwargs):
        """
        The submit function is a wrapper around the executor's submit method.
        It acquires the semaphore, which limits the number of concurrent threads.
        If you exceed that limit, your thread will be blocked until one of the other
        threads completes.

        :param self: Access the attributes and methods of the class in python
        :param fn: Pass the function to be executed
        :param *args: Pass a non-keyworded, variable-length argument list
        :param **kwargs: Pass a variable-length list of arguments to the function
        :return: A future object
        :doc-author: Trelent
        """
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except Exception as exc:
            self.semaphore.release()
            raise exc
        else:
            future.add_done_callback(lambda x: self.semaphore.release())
            return future

    """See concurrent.futures.Executor#shutdown"""

    def shutdown(self, wait=False):
        self.executor.shutdown(wait)

    def addFuture(self, nme, fn, *args, **kwargs):
        sleep(self.buffer)
        self.futures[self.submit(fn, *args, **kwargs)] = nme

    def check_futures(self):
        """
        The check_futures function is a helper function that checks the results of
        as_completed() to see if any exceptions were raised. If an exception was
        raised, it is logged and the thread fails. Otherwise, it returns a dictionary
        of all results from as_completed(). This dictionary is then passed to the
        check_results function.

        :param self: Reference the object itself
        :return: A dictionary of the results from the futures that have completed
        :doc-author: Trelent
        """
        fails = 0
        results = dict()
        for future in as_completed(self.futures):
            obj = self.futures[future]
            try:
                results[obj] = future.result()
            except Exception as exc:
                err = f"({self.tag}): {obj} generated an exception:\n{dashes}\n"
                err += err_catch(exc)
                print(err)
                fails += 1
        if fails > 0:
            raise ThreaderFailed(self.tag, fails)
        else:
            return dict(sorted(results.items(), key=operator.itemgetter(0)))
