class Act(ABC):
    @abstractmethod
    def __call__(self, state: ImplicitState, memory: MutableMapping) -> Tuple[str, Dict[str, Any], Any]:
        """Return (tool_id, params, observation)."""
        ...