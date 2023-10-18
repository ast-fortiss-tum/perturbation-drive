import logging
import csv


class CSVLogHandler(logging.FileHandler):
    """
    Util class to log perturbation output and metrics

    :param: filename="logs.csv": String name of log file
    :param: mode="w": Mode of the logger. Here we can use options such as "w", "a", ...
    :param: encoding=None: Encoding of the file
    """

    def __init__(self, filename="logs.csv", mode="w", encoding=None):
        super().__init__(filename, mode, encoding, delay=False)
        self.writer = csv.writer(
            self.stream, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        self.current_row = []

    def emit(self, record):
        if isinstance(record.msg, (list, tuple)):
            self.current_row.extend(record.msg)
        else:
            self.current_row.append(record.msg)

    def flush_row(self):
        if self.current_row:
            self.writer.writerow(self.current_row)
            self.flush()
            self.current_row = []

