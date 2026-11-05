from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import GraphStorage
from omnimembench.engine.operation.Recall import NeuralGraphRecall
from omnimembench.engine.operation.Store import NeuralGraphStore
from omnimembench.engine.function.MultiModalRetrieval import MultiModalRetrieval
from omnimembench.engine.utils.Display import *

class NeuralGraph(ExplicitMemory):
    """
    NeuralGraph (Neural Graph Memory): 
    
    Reference: Neural Graph Memory: A Structured Approach to Long-Term Memory in Multimodal Agents
    https://www.researchgate.net/profile/Matt-Fisher-7/publication/394440420_Neural_Graph_Memory_A_Structured_Approach_to_Long-Term_Memory_in_Multimodal_Agents/links/689ab8c337b271210509c20f/Neural-Graph-Memory-A-Structured-Approach-to-Long-Term-Memory-in-Multimodal-Agents.pdf
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        # Use GraphStorage for graph-structured memory
        self.storage = GraphStorage(self.config.args.storage)
        
        # Create multimodal retrieval
        self.multimodal_retrieval = MultiModalRetrieval(self.config.args.multimodal_retrieval)
        
        # Create store and recall operations
        self.store_op = NeuralGraphStore(
            self.config.args.store, 
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
        
        self.recall_op = NeuralGraphRecall(
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
        Store observation (text, image, or both) as a graph node
        """
        self.store_op(observation)
    
    def recall(self, query) -> object:
        """
        Recall relevant memories using query-aware graph traversal
        """
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.node_counter)

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        return {
            'node': self.storage.node,
            'edge': self.storage.edge,
            'node_counter': self.storage.node_counter,
            'edge_counter': self.storage.edge_counter,
            'memory_order_map': self.storage.memory_order_map,
            'mm_tensorstore': self.multimodal_retrieval.tensorstore,
            'mm_memory_metadata': self.multimodal_retrieval.memory_metadata,
        }

    def _set_state(self, state):
        self.storage.node = state['node']
        self.storage.edge = state['edge']
        self.storage.node_counter = state['node_counter']
        self.storage.edge_counter = state['edge_counter']
        self.storage.memory_order_map = state['memory_order_map']
        self.multimodal_retrieval.tensorstore = state['mm_tensorstore']
        self.multimodal_retrieval.memory_metadata = state['mm_memory_metadata']