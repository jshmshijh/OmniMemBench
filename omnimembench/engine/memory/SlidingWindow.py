from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import LinearStorage
from omnimembench.engine.operation.Recall import SlidingWindowRecall
from omnimembench.engine.operation.Store import SlidingWindowStore
from omnimembench.engine.utils.Display import *

class SlidingWindow(ExplicitMemory):
    """
    SlidingWindow (Short-term Memory, FIFO): Maintain the most recent information and concatenate them into one string as the context.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.recall_op = SlidingWindowRecall(self.config.args.recall, storage = self.storage)
        self.store_op = SlidingWindowStore(self.config.args.store, storage = self.storage, time_retrieval = self.recall_op.time_retrieval)

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage
        })

    def reset(self):
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, query) -> object:
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
    
    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        return {
            'storage_memory_list': self.storage.memory_list,
            'storage_counter': self.storage.counter,
            'time_tensorstore': self.recall_op.time_retrieval.tensorstore,
        }

    def _set_state(self, state):
        self.storage.memory_list = state['storage_memory_list']
        self.storage.counter = state['storage_counter']
        self.recall_op.time_retrieval.tensorstore = state['time_tensorstore']