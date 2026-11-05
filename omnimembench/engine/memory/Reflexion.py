from omnimembench.engine.memory.BaseMemory import ExplicitMemory
from omnimembench.engine.utils.Storage import LinearStorage
from omnimembench.engine.operation.Recall import ReflexionRecall
from omnimembench.engine.operation.Store import FUMemoryStore
from omnimembench.engine.operation.Optimize import ReflexionOptimize
from omnimembench.engine.utils.Display import *

class Reflexion(ExplicitMemory):
    """
    Reflexion [1]: A famous memory method that can learn to memorize from previous trajectories by optimization.

    [1] Shinn, Noah, et al. "Reflexion: Language agents with verbal reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).
    https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf
    """
    def __init__(self, config):
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.insight = {'global_insight': ''}

        self.store_op = FUMemoryStore(self.config.args.store, storage = self.storage)
        self.recall_op = ReflexionRecall(self.config.args.recall, storage = self.storage, insight = self.insight)
        self.optimize_op = ReflexionOptimize(self.config.args.optimize, insight = self.insight)

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage,
            'Insight': self.insight
        })

    def reset(self):
        self.insight = {'global_insight': ''}
        self.__reset_objects__([self.storage, self.store_op, self.recall_op, self.optimize_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, query) -> object:
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        self.optimize_op(**kwargs)

    def _get_state(self):
        return {
            'storage_memory_list': self.storage.memory_list,
            'storage_counter': self.storage.counter,
            'insight': dict(self.insight),
        }

    def _set_state(self, state):
        self.storage.memory_list = state['storage_memory_list']
        self.storage.counter = state['storage_counter']
        self.insight.clear()
        self.insight.update(state['insight'])


