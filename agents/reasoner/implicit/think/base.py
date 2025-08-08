class Think(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> str:
        ...
