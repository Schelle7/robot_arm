import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class ScalarWriter:
    def __init__(self, log_dir: str):
        self.writer = EventFileWriter(log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        self.writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
