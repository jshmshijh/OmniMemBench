from abc import ABC, abstractmethod
import pickle
import os


class BaseMemory(ABC):
    """
    Basic class for memory methods.
    """
    def __init__(self, config) -> None:
        self.config = config

    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def store(self, obj) -> None:
        pass
    
    @abstractmethod
    def recall(self, obj) -> object:
        pass
    
    @abstractmethod
    def manage(self, operation, **kwargs) -> None:
        pass
    
    @abstractmethod
    def optimize(self, **kwargs) -> None:
        pass

    def save_state(self, path):
        """
        Serialize memory state to disk.
        Subclasses should override _get_state() to specify what to save.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = self._get_state()
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, path):
        """
        Restore memory state from disk.
        Subclasses should override _set_state() to specify how to restore.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._set_state(state)

    def _get_state(self):
        """Return a picklable dict of memory state. Override in subclass."""
        raise NotImplementedError(f"{type(self).__name__} must implement _get_state()")

    def _set_state(self, state):
        """Restore memory state from dict. Override in subclass."""
        raise NotImplementedError(f"{type(self).__name__} must implement _set_state()")

class ExplicitMemory(BaseMemory):
    """
    Explicit memory indicates the methods that represent memory contents with texts.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
    
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def store(self, observation) -> None:
        pass
    
    @abstractmethod
    def recall(self, query) -> str:
        pass

    @abstractmethod
    def display(self) -> None:
        pass
    
    @abstractmethod
    def manage(self, operation, **kwargs) -> None:
        pass
    
    @abstractmethod
    def optimize(self, **kwargs) -> None:
        pass

class ImplicitMemory(BaseMemory):
    """
    Implicit memory indicates the methods that represent memory contents with parameters.
    """
    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def store(self, obj) -> None:
        pass
    
    @abstractmethod
    def recall(self, query) -> object:
        pass
    
    @abstractmethod
    def manage(self, operation, **kwargs) -> None:
        pass
    
    @abstractmethod
    def optimize(self, **kwargs) -> None:
        pass