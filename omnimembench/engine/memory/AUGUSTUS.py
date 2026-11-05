from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import TagGraphStorage
from omnimembench.engine.operation.Recall import AUGUSTUSRecall
from omnimembench.engine.operation.Store import AUGUSTUSStore
from omnimembench.engine.function.MultiModalRetrieval import MultiModalRetrieval
from omnimembench.engine.function.ConceptExtractor import LLMConceptExtractor, SimpleConceptExtractor
from omnimembench.engine.function.ConceptBasedRetrieval import ConceptBasedRetrieval
from omnimembench.engine.utils.Display import *

class AUGUSTUS(ExplicitMemory):
    """
    A multimodal memory system with concept-based retrieval and contextualized graph memory.
        
    Reference: AUGUSTUS: An LLM-Driven Multimodal Agent System with Contextualized User Memory
    https://arxiv.org/pdf/2510.15261
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        # Contextual Memory: Graph-structured memory with concept tags
        # As per AUGUSTUS paper: arangoDB for contextual memory
        # Using TagGraphStorage as an in-memory equivalent
        self.contextual_memory = TagGraphStorage(self.config.args.storage)
        
        # Create multimodal retrieval
        self.multimodal_retrieval = MultiModalRetrieval(self.config.args.multimodal_retrieval)
        
        # Create concept extractor
        extractor_method = getattr(self.config.args.concept_extractor, 'method', 'SimpleConceptExtractor')
        if extractor_method == 'LLMConceptExtractor':
            self.concept_extractor = LLMConceptExtractor(self.config.args.concept_extractor)
        elif extractor_method == 'SimpleConceptExtractor':
            self.concept_extractor = SimpleConceptExtractor(self.config.args.concept_extractor)
        else:
            raise ValueError(f"Unsupported concept extractor method: {extractor_method}")
        
        # Create concept-based retrieval
        self.concept_retrieval = ConceptBasedRetrieval(self.config.args.concept_retrieval)
        self.concept_retrieval.set_storage(self.contextual_memory)
        
        # Create store and recall operations
        self.store_op = AUGUSTUSStore(
            self.config.args.store, 
            contextual_memory=self.contextual_memory,
            multimodal_retrieval=self.multimodal_retrieval,
            concept_extractor=self.concept_extractor
        )
        
        self.recall_op = AUGUSTUSRecall(
            self.config.args.recall, 
            contextual_memory=self.contextual_memory,
            multimodal_retrieval=self.multimodal_retrieval,
            concept_extractor=self.concept_extractor,
            concept_retrieval=self.concept_retrieval
        )

        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display, 
            register_dict={
                'Contextual Memory': self.contextual_memory
            }
        )
    
    def reset(self) -> None:
        self.__reset_objects__([
            self.contextual_memory, 
            self.store_op, 
            self.recall_op, 
            self.concept_extractor, 
            self.concept_retrieval
        ])

    def store(self, observation) -> None:
        """
        Store observation in Contextual Memory (concept-tagged graph).
        """
        self.store_op(observation)
    
    def recall(self, query) -> object:
        """
        Recall relevant memories using CoPe search from Contextual Memory.
        """
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.contextual_memory.node_counter)
    
    def get_concept_statistics(self) -> dict:
        """
        Get statistics about concepts in the contextual memory graph.
        
        Returns:
            dict: Statistics including total concepts, most common concepts, etc.
        """
        return self.contextual_memory.get_concept_statistics()
    
    @property
    def storage(self):
        """
        Backward compatibility: return contextual_memory as storage.
        This allows existing code that references self.storage to continue working.
        """
        return self.contextual_memory

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        return {
            'node': self.contextual_memory.node,
            'edge': self.contextual_memory.edge,
            'node_counter': self.contextual_memory.node_counter,
            'edge_counter': self.contextual_memory.edge_counter,
            'memory_order_map': self.contextual_memory.memory_order_map,
            'concept_index': self.contextual_memory.concept_index,
            'node_concepts': self.contextual_memory.node_concepts,
            'mm_tensorstore': self.multimodal_retrieval.tensorstore,
            'mm_memory_metadata': self.multimodal_retrieval.memory_metadata,
        }

    def _set_state(self, state):
        self.contextual_memory.node = state['node']
        self.contextual_memory.edge = state['edge']
        self.contextual_memory.node_counter = state['node_counter']
        self.contextual_memory.edge_counter = state['edge_counter']
        self.contextual_memory.memory_order_map = state['memory_order_map']
        self.contextual_memory.concept_index = state['concept_index']
        self.contextual_memory.node_concepts = state['node_concepts']
        self.multimodal_retrieval.tensorstore = state['mm_tensorstore']
        self.multimodal_retrieval.memory_metadata = state['mm_memory_metadata']

