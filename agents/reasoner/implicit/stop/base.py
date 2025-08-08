class StopCondition(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Optional[str]:
        ...
