from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.UniversalRAGStorage import UniversalRAGStorage
from omnimembench.engine.operation.Recall import UniversalRAGRecall
from omnimembench.engine.operation.Store import UniversalRAGStore
from omnimembench.engine.function.UniversalRAGRouting import UniversalRAGRouting
from omnimembench.engine.function.UniversalRAGRetrieval import UniversalRAGRetrieval
from omnimembench.engine.utils.Display import *

class UniversalRAG(ExplicitMemory):
    """
    UniversalRAG Memory
    
    Features:
    - Dynamic routing: Automatically routes queries to 'no', 'document', or 'image'
    - Multi-modal retrieval: Supports text and image retrieval using GME embeddings
    - Unified interface: Compatible with existing memory framework

    Reference: UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities
    https://arxiv.org/pdf/2504.20734
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.storage = UniversalRAGStorage(self.config.args.storage)
        
        self.routing = UniversalRAGRouting(self.config.args.routing)
        
        self.retrieval = UniversalRAGRetrieval(
            self.config.args.retrieval,
            storage=self.storage
        )
        
        self.store_op = UniversalRAGStore(
            self.config.args.store,
            storage=self.storage
        )
        
        self.recall_op = UniversalRAGRecall(
            self.config.args.recall,
            storage=self.storage,
            routing=self.routing,
            retrieval=self.retrieval
        )
        
        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display,
            register_dict={'Memory Storage': self.storage}
        )
    
    def reset(self) -> None:
        self.__reset_objects__([
            self.storage, self.store_op, self.recall_op,
            self.routing, self.retrieval
        ])
    
    def store(self, observation) -> None:
        """
        Store observation (text, image, or both) with automatic feature extraction.
        """
        self.store_op(observation)
    
    def recall(self, query) -> object:
        """
        Recall relevant memories using dynamic routing and multi-modal retrieval.
        Routes query to appropriate modality ('no', 'document', or 'image') and retrieves accordingly.
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
            'document_index': self.storage.document_index,
            'image_index': self.storage.image_index,
            'audio_index': self.storage.audio_index,
            'video_index': self.storage.video_index,
        }

    def _set_state(self, state):
        self.storage.memory_list = state['storage_memory_list']
        self.storage.counter = state['storage_counter']
        self.storage.document_index = state['document_index']
        self.storage.image_index = state['image_index']
        self.storage.audio_index = state['audio_index']
        self.storage.video_index = state['video_index']
