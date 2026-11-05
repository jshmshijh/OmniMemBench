from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import LinearStorage
from omnimembench.engine.operation.Recall import MultimodalRAGRecall
from omnimembench.engine.operation.Store import MultimodalRAGStore
from omnimembench.engine.function.MultiModalRetrieval import MultiModalRetrieval
from omnimembench.engine.utils.Display import *

class MultimodalRAG(ExplicitMemory):
    """
    MultimodalRAG: 
    Supports text, image, and text+image storage and retrieval using multimodal embeddings.

    Based on the retrieval format in paper: 
    MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text
    https://aclanthology.org/2022.emnlp-main.375.pdf
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.storage = LinearStorage(self.config.args.storage)
        
        self.multimodal_retrieval = MultiModalRetrieval(self.config.args.multimodal_retrieval)
        
        self.store_op = MultimodalRAGStore(
            self.config.args.store, 
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
        
        self.recall_op = MultimodalRAGRecall(
            self.config.args.recall, 
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )

        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display, 
            register_dict={'Memory Storage': self.storage}
        )
    
    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        """
        Store observation (text, image, or both)
        """
        self.store_op(observation)
    
    def recall(self, query) -> str:
        """
        Recall relevant memories based on query
        """
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
            'mm_tensorstore': self.multimodal_retrieval.tensorstore,
            'mm_memory_metadata': self.multimodal_retrieval.memory_metadata,
        }

    def _set_state(self, state):
        self.storage.memory_list = state['storage_memory_list']
        self.storage.counter = state['storage_counter']
        self.multimodal_retrieval.tensorstore = state['mm_tensorstore']
        self.multimodal_retrieval.memory_metadata = state['mm_memory_metadata']