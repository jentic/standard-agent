class DecidePolicy(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> str:
        """Return one of: "REASON" | "TOOL" | "HALT"."""
        ...