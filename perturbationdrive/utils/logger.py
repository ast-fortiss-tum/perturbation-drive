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

    def emit(self, record):
        if isinstance(record.msg, (list, tuple)):
            self.writer.writerow(record.msg)
        else:
            self.writer.writerow([record.msg])
        self.flush()
