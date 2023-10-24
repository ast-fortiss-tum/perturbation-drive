from collections import deque


class CircularBuffer(deque):
    """
    Cicular buffer which stores `size` elements
    """

    def __init__(self, size):
        """
        Inits an circular buffer with a maximum of size elements
        """
        super().__init__(maxlen=size)

    def add(self, value):
        """
        Adds an element to the circular buffer
        """
        rounded_tuple = tuple(round(num, 2) for num in value)
        self.append(rounded_tuple)

    def all_elements_equal(self):
        """
        Checks if all elements are equal in the ring buffer
        """
        return len(set(self)) == 1

    def length(self):
        """
        Returns the length of the crash buffer
        """
        return len(self)
