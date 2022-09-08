from abc import ABC, abstractmethod


class Format(ABC):
    @abstractmethod
    def parse_format(self, file):
        pass
