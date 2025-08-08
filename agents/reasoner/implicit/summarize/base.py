class Summarizer(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState) -> str:
        ...
