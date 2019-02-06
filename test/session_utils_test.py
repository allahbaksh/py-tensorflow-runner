import math
import random
from threading import Thread
from time import sleep

from py_pipe.pipe import Pipe

from py_tensorflow_runner.session_utils import SessionRunner, SessionRunnable, Inference
import tensorflow as tf


class SessionTest:

    def __init__(self, flush_pipe_on_read=False):
        self.__thread = None
        self.__flush_pipe_on_read = flush_pipe_on_read
        self.__run_session_on_thread = False
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

    def __in_pipe_process(self, inference):
        a, b = inference.get_input()
        inference.set_data([a*a, b*b])
        return inference

    def __out_pipe_process(self, result):
        result, inference = result
        inference.set_result(math.sqrt(result))
        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def use_threading(self, run_on_thread=True):
        self.__run_session_on_thread = run_on_thread

    def use_session_runner(self, session_runner):
        self.__session_runner = session_runner
        self.__tf_sess = session_runner.get_session()

        self.__x = tf.placeholder(tf.int32, None)
        self.__y = tf.placeholder(tf.int32, None)
        self.__z = tf.add(self.__x, self.__y)

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            self.__in_pipe.pull_wait()
            ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__session_runner.get_in_pipe().push(
                    SessionRunnable(self.__job, inference, run_on_thread=self.__run_session_on_thread))

    def __job(self, inference):
        self.__out_pipe.push(
            (self.__tf_sess.run(self.__z,
                                feed_dict={self.__x: inference.get_data()[0], self.__y: inference.get_data()[1]}),
             inference))

if __name__ == '__main__':
    session_runner = SessionRunner()
    session_runner.start()

    addOnGPU = SessionTest()

    ip = addOnGPU.get_in_pipe()
    op = addOnGPU.get_out_pipe()

    addOnGPU.use_session_runner(session_runner)
    addOnGPU.run()


    def send():
        while True:
            ip.push_wait()
            inference = Inference([random.randint(0, 100), random.randint(0, 100)])
            ip.push(inference)
            sleep(1)


    def receive():
        while True:
            op.pull_wait()
            ret, inference = op.pull()
            if ret:
                print('result('+str(inference.get_input())+') = ' + str(inference.get_result()))


    Thread(target=send).start()
    Thread(target=receive).start()
