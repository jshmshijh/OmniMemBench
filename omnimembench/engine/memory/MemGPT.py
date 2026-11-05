from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import LinearStorage
from omnimembench.engine.operation.Recall import MemGPTRecall
from omnimembench.engine.operation.Store import MemGPTStore
from omnimembench.engine.utils.Display import *

class MemGPT(ExplicitMemory):
    """
    MemGPT [1]: A hierarchical memory model that treat the memory system as an operation system.

    [1] Packer, Charles, et al. "Memgpt: Towards llms as operating systems." arXiv preprint arXiv:2310.08560 (2023).
    https://arxiv.org/pdf/2310.08560
    """
    def __init__(self, config):
        super().__init__(config)

        self.main_context = {
            'working_context': LinearStorage(self.config.args.storage),
            'FIFO_queue': LinearStorage(self.config.args.storage),
            'recursive_summary': {'global': 'None'}
        }
        self.recall_storage = LinearStorage(self.config.args.storage)
        self.archival_storage = LinearStorage(self.config.args.storage)

        self.recall_op = MemGPTRecall(
            self.config.args.recall,
            main_context = self.main_context,
            recall_storage = self.recall_storage,
            archival_storage = self.archival_storage
        )
        self.store_op = MemGPTStore(
            self.config.args.store,
            main_context = self.main_context,
            recall_storage = self.recall_storage,
            recall_retrieval = self.recall_op.recall_retrieval,
            truncation = self.recall_op.truncation
        )

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Working Memory': self.main_context['working_context'],
            'Recursive Memory Summary': self.main_context['recursive_summary'],
            'FIFO Memory': self.main_context['FIFO_queue'],
            'Recall Storage': self.recall_storage,
            'Archival Storage': self.archival_storage
        })
        
    def reset(self) -> None:
        self.main_context = {
            'working_context': LinearStorage(self.config.args.storage),
            'FIFO_queue': LinearStorage(self.config.args.storage),
            'recursive_summary': {'global': 'None'}
        }
        self.__reset_objects__([self.recall_storage, self.archival_storage, self.recall_op, self.store_op])
    
    def store(self, observation) -> None:
        self.store_op(observation)

    def recall(self, observation) -> object:
        return self.recall_op(observation)
    
    def display(self) -> None:
        self.auto_display('-')

    def manage(self) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        return {
            'wc_memory_list': self.main_context['working_context'].memory_list,
            'wc_counter': self.main_context['working_context'].counter,
            'fifo_memory_list': self.main_context['FIFO_queue'].memory_list,
            'fifo_counter': self.main_context['FIFO_queue'].counter,
            'recursive_summary': self.main_context['recursive_summary'],
            'recall_storage_memory_list': self.recall_storage.memory_list,
            'recall_storage_counter': self.recall_storage.counter,
            'recall_tensorstore': self.recall_op.recall_retrieval.tensorstore,
        }

    def _set_state(self, state):
        self.main_context['working_context'].memory_list = state['wc_memory_list']
        self.main_context['working_context'].counter = state['wc_counter']
        self.main_context['FIFO_queue'].memory_list = state['fifo_memory_list']
        self.main_context['FIFO_queue'].counter = state['fifo_counter']
        self.main_context['recursive_summary'] = state['recursive_summary']
        self.recall_storage.memory_list = state['recall_storage_memory_list']
        self.recall_storage.counter = state['recall_storage_counter']
        self.recall_op.recall_retrieval.tensorstore = state['recall_tensorstore']
        # store_op shares the same recall_retrieval reference
        self.store_op.recall_retrieval.tensorstore = state['recall_tensorstore']