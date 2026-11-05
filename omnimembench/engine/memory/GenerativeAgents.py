from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import LinearStorage
from omnimembench.engine.operation.Recall import GenerativeAgentsRecall
from omnimembench.engine.operation.Store import GenerativeAgentsStore
from omnimembench.engine.operation.Reflect import GenerativeAgentsReflect
from omnimembench.engine.utils.Display import *

class GenerativeAgents(ExplicitMemory):
    """
    GenerativeAgents [1]: A pioneer memory model with weighted retrieval combination and self-reflection mechanism.
    
    [1] Park, Joon Sung, et al. "Generative agents: Interactive simulacra of human behavior." Proceedings of the 36th annual acm symposium on user interface software and technology. 2023.
    https://dl.acm.org/doi/pdf/10.1145/3586183.3606763
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.recall_op = GenerativeAgentsRecall(self.config.args.recall, storage = self.storage)
        self.reflect_op = GenerativeAgentsReflect(
            self.config.args.reflect,
            storage = self.storage,
            text_retrieval = self.recall_op.text_retrieval,
            time_retrieval = self.recall_op.time_retrieval
        )
        self.store_op = GenerativeAgentsStore(
            self.config.args.store,
            storage = self.storage,
            text_retrieval = self.recall_op.text_retrieval,
            time_retrieval = self.recall_op.time_retrieval,
            importance_retrieval = self.recall_op.importance_retrieval,
            imporatance_judge = self.recall_op.imporatance_judge,
            reflector = self.reflect_op.reflector
        )

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage
        })
    
    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.reflect_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, query) -> object:
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
        
    def manage(self, operation, **kwargs) -> None:
        if operation == 'reflect':
            insight_list = self.reflect_op(**kwargs)
            for insight in insight_list:
                self.store_op(insight)
        else:
            raise "Management error."
    
    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        return {
            'storage_memory_list': self.storage.memory_list,
            'storage_counter': self.storage.counter,
            'text_tensorstore': self.recall_op.text_retrieval.tensorstore,
            'time_tensorstore': self.recall_op.time_retrieval.tensorstore,
            'importance_tensorstore': self.recall_op.importance_retrieval.tensorstore,
            'accmulated_importance': self.reflect_op.reflector.accmulated_importance,
            'recursion_list': self.reflect_op.reflector.recursion_list,
        }

    def _set_state(self, state):
        self.storage.memory_list = state['storage_memory_list']
        self.storage.counter = state['storage_counter']
        self.recall_op.text_retrieval.tensorstore = state['text_tensorstore']
        self.recall_op.time_retrieval.tensorstore = state['time_tensorstore']
        self.recall_op.importance_retrieval.tensorstore = state['importance_tensorstore']
        self.reflect_op.reflector.accmulated_importance = state['accmulated_importance']
        self.reflect_op.reflector.recursion_list = state['recursion_list']