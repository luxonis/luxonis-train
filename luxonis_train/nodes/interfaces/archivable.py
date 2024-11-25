from abc import ABC, abstractmethod

class Archivable(ABC):
    @abstractmethod
    def get_metadata(self) -> dict:
        ...