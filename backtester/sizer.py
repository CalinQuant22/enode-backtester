import abc



class BaseSizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def size_order(self, portfolio, signal_event) -> int:
        raise NotImplementedError()


# for now we only use basic logic we are gonna need to implement more sophisticated logic later
class FixedSizeSizer(BaseSizer):
    def __init__(self, default_size: int = 50):
        self.default_size = default_size

    def size_order(self, portfolio, signal_event) -> int:
        return self.default_size


