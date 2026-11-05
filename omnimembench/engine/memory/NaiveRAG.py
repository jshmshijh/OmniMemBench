from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import LinearStorage
from omnimembench.engine.operation.Recall import NaiveRAGRecall
from omnimembench.engine.operation.Store import NaiveRAGStore
from omnimembench.engine.utils.Display import *

class NaiveRAG(ExplicitMemory):
    """
    NaiveRAG: Calculate semantic similarities with text embeddings to retrieval most relevant information.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.storage = LinearStorage(self.config.args.storage)
        self.recall_op = NaiveRAGRecall(self.config.args.recall, storage = self.storage)
        self.store_op = NaiveRAGStore(self.config.args.store, storage = self.storage, text_retrieval = self.recall_op.text_retrieval)

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage
        })

    def reset(self):
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, observation) -> object:
        return self.recall_op(observation)
    
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
            'text_tensorstore': self.recall_op.text_retrieval.tensorstore,
        }

    def _set_state(self, state):
        self.storage.memory_list = state['storage_memory_list']
        self.storage.counter = state['storage_counter']
        self.recall_op.text_retrieval.tensorstore = state['text_tensorstore']
