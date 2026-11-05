from abc import ABC, abstractmethod
import threading
from omnimembench.engine.function import *

def __recall_convert_str_to_observation__(method):
    """
    If the input is a string, convert it to the dict form.
    """
    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {'text': observation})
        else:
            return method(self, observation)
    return wrapper


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_eval_data(memories):
    """Extract dialogue_ids and texts from a list of memory elements.
    For multimodal elements, append caption info so retrieval judge can see full content.
    """
    ids, texts = [], []
    for mem in memories:
        if isinstance(mem, dict):
            did = mem.get('dialogue_id', '')
            if did:
                ids.append(did)
            text = mem.get('text', '')
            # Append multimodal captions (consistent with text-mode memory_store behaviour)
            for modality, id_key, cap_key in [
                ('image', 'img_id', 'caption'),
                ('audio', 'audio_id', 'caption'),
                ('video', 'video_id', 'caption'),
            ]:
                info = mem.get(modality)
                if info and isinstance(info, dict):
                    mid_val = info.get(id_key, '')
                    mcap = info.get(cap_key, '')
                    if mid_val or mcap:
                        text += f'\n{modality}:\n{modality}_id: {mid_val}\n{modality}_caption: {mcap}'
            texts.append(text)
        else:
            texts.append(str(mem))
    return ids, texts


def _get_context_topk(config):
    """Read context_topk from recall config; None means use all."""
    return getattr(config, 'context_topk', None)


# ── Base ─────────────────────────────────────────────────────────────────────

class BaseRecall(ABC):
    def __init__(self, config):
        self.config = config
    
    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()
    
    @abstractmethod
    def reset(self):
        pass

    @ abstractmethod
    def __call__(self, query):
        pass

class FUMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        # No retrieval — mark empty for eval
        self._local.last_retrieved_ids = []
        self._local.last_retrieved_texts = []

        if self.storage.is_empty():
            return self.config.empty_memory
        # Pass complete memory elements (including timestamp) instead of just text
        memory_context = self.utilization(self.storage.get_all_memory_in_order())
        return self.truncation(memory_context)

class MMFUMemoryRecall(BaseRecall):
    """
    MultiModal Full Memory Recall:
    Retrieves all memories (text + images) and truncates based on token budget.
    Unlike FUMemoryRecall which only handles text, this preserves multimodal data.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Get all memories in order (includes both text and image data)
        all_memories = self.storage.get_all_memory_in_order()
        
        # Apply multimodal truncation (based on token budget: text + image tokens)
        truncated_memories = self.truncation(all_memories)
        
        # Expose retrieved ids/texts for evaluation
        self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(truncated_memories)
        
        # Use MultiModalUtilization to format the result
        result = self.utilization(truncated_memories)
        
        return result

class SlidingWindowRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.time_retrieval])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Get the most recent information.
        ranking_ids = self.time_retrieval(query['text'])
        # Expose retrieved ids for evaluation
        try:
            elements = [self.storage.get_memory_element_by_mid(mid) for mid in ranking_ids]
            self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(elements)
        except Exception:
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []

        # context_topk: only pass a subset to LLM
        context_topk = _get_context_topk(self.config)
        context_elements = elements[:context_topk] if context_topk else elements

        # Pass complete memory elements (including timestamp) instead of just text
        memory_context = self.utilization(context_elements)

        return self.truncation(memory_context)


class NaiveRAGRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Retrieval process.
        ranking_ids = self.text_retrieval(query['text'])
        elements = [self.storage.get_memory_element_by_mid(mid) for mid in ranking_ids]

        # Store ALL retrieved for eval
        self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(elements)

        # context_topk: only pass a subset to LLM
        context_topk = _get_context_topk(self.config)
        context_elements = elements[:context_topk] if context_topk else elements

        memory_context = self.utilization(context_elements)
        return self.truncation(memory_context)


class GenerativeAgentsRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)
        self.importance_retrieval = eval(self.config.importance_retrieval.method)(self.config.importance_retrieval)
        
        self.imporatance_judge = eval(self.config.importance_judge.method)(self.config.importance_judge)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval, self.time_retrieval, self.importance_retrieval, self.imporatance_judge])

    def __retention__(self, target_ids, timestamp):
        """Update the memory of `target_ids` with their recency to `timestamp`."""
        for index in target_ids:
            self.time_retrieval.update(index, timestamp)

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        text = query['text']
        # Prefer timestamp field, but time_retrieval needs numeric values
        if 'timestamp' in query:
            if 'time' in query:
                timestamp = query['time']
            else:
                timestamp = self.storage.counter
        elif 'time' not in query:
            timestamp = self.storage.counter
        else:
            timestamp = query['time']
        
        # Calculate weighted retrieval scores.
        text_scores, _ = self.text_retrieval(text, topk=False, with_score = True, sort = False)
        recency_scores, _ = self.time_retrieval(timestamp, topk=False, with_score = True, sort = False)
        importance_scores, _ = self.importance_retrieval(None, topk=False, with_score = True, sort = False)

        score_metrix = torch.stack([text_scores, recency_scores, importance_scores], dim=1)
        comprehensive_scores = torch.matmul(score_metrix, torch.ones(3).to(self.text_retrieval.encoder.device))

        scores, ranking_ids = torch.sort(comprehensive_scores, descending=True)

        if hasattr(self.config, 'topk'):
            scores, ranking_ids = scores[:self.config.topk], ranking_ids[:self.config.topk]
        
        # Update the recency of retrieved memory.
        self.__retention__(ranking_ids, timestamp)

        # Collect all retrieved elements and expose for eval
        try:
            elements = [self.storage.get_memory_element_by_mid(int(mid)) for mid in ranking_ids]
            self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(elements)
        except Exception:
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            elements = []

        # context_topk: only pass a subset to LLM
        context_topk = _get_context_topk(self.config)
        context_elements = elements[:context_topk] if context_topk else elements

        memory_context = self.utilization(context_elements)
        return self.truncation(memory_context)


class MemGPTRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

        self.main_context = kwargs['main_context']
        self.recall_storage = kwargs['recall_storage']
        self.archival_storage = kwargs['archival_storage']

        self.recall_retrieval = eval(self.config.recall_retrieval.method)(self.config.recall_retrieval)
        self.archival_retrieval = eval(self.config.archival_retrieval.method)(self.config.archival_retrieval)
        self.trigger = LLMTrigger(self.config.trigger)
        self.warning_number = int(self.config.warning_threshold * self.config.truncation.number)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.recall_retrieval, self.archival_retrieval, self.trigger])

    def __get_current_memory_prompt(self):
        """
        To provide the current state of memory model for constructing prompts.
        """
        piece_num, piece_mode = self.truncation.get_piece_number(self.get_current_memory_context()), self.config.truncation.mode
        working_memory = '\n'.join(['[%d] %s' % (wid, wtext['text']) for wid, wtext in enumerate(self.main_context['working_context'].get_all_memory_in_order())])
        FIFO_memory = '\n'.join(['[%d] %s' % (fid, ftext['text']) for fid, ftext in enumerate(self.main_context['FIFO_queue'].get_all_memory_in_order())])
        prompt = """Total Capacity: %s %ss
Working Memory(capacity %s words):
%s
Recursive Memory Summary:
%s
FIFO Memory:
%s
""" % (piece_num, piece_mode, self.config.truncation.number, working_memory, self.main_context['recursive_summary']['global'], FIFO_memory)

        return prompt

    def get_current_memory_context(self):
        # Pass complete memory elements (including timestamp) instead of just text
        memory_context = self.utilization({
            'Working Memory': self.main_context['working_context'].get_all_memory_in_order(),
            'Recursive Memory Summary': self.main_context['recursive_summary']['global'],
            'FIFO Memory': self.main_context['FIFO_queue'].get_all_memory_in_order()
        })
        return memory_context

    def __trigger_function_call__(self, text):
        """Trigger extensible functions."""
        warning_flag = self.truncation.get_piece_number(text) > self.warning_number

        info_dict = {}
        
        info_dict['text'] = text
        info_dict['memory_prompt'] = self.__get_current_memory_prompt()

        # Insert warning sentences.
        if warning_flag:
            info_dict['warning_content'] = self.config.warning_content + '\n'
        else:
            info_dict['warning_content'] = ''

        # Allow trigger none function.
        if hasattr(self.config.trigger, 'no_execute'):
            info_dict['no_execute_prompt'] = self.config.trigger.no_execute + '\n'
        else:
            info_dict['no_execute_prompt'] = ''

        # Add few-shot examples.
        info_dict['few_shot'] = self.config.trigger.few_shot

        # Get trigger results, parse and execute them.
        exe_list = self.trigger(info_dict)
        for (func_name, func_args) in exe_list:
            try:
                execute_command = 'self.__%s__' % func_name
                eval(execute_command)(*func_args)
                print('Successfully execute Function [%s(%s)]' % (func_name, func_args))
            except Exception as e:
                print('Fail to execute Function [%s(%s)]: %s'% (func_name, func_args, e))

    def __memory_retrieval__(self, query):
        if self.archival_storage.is_empty():
            return self.config.empty_memory
        text = query['text']
        ranking_ids = self.archival_retrieval(text)
        for mid in ranking_ids:
            self.main_context['working_context'].add(self.archival_storage.get_memory_element_by_mid(mid))
    
    def __memory_recall__(self, query):
        if self.recall_storage.is_empty():
            return self.config.empty_memory
        text = query['text']
        ranking_ids = self.recall_retrieval(text)
        for mid in ranking_ids:
            self.main_context['FIFO_queue'].add(self.recall_storage.get_memory_element_by_mid(mid))

    def __memory_archive__(self, memory_list):
        for mid in memory_list:
            element = self.main_context['FIFO_queue'].get_memory_element_by_mid(mid)
            self.recall_storage.add(element)
            self.recall_retrieval.add(element['text'])

        self.main_context['FIFO_queue'].delete_by_mid_list(memory_list)
    
    def __memory_transfer__(self, memory_list):
        for mid in memory_list:
            element = self.main_context['FIFO_queue'].get_memory_element_by_mid(mid)
            self.main_context['working_context'].add(element)

        self.main_context['FIFO_queue'].delete_by_mid_list(memory_list)
    
    def __memory_save__(self, memory_list):
        for mid in memory_list:
            element = self.main_context['working_context'].get_memory_element_by_mid(mid)
            self.archival_storage.add(element)
            self.archival_retrieval.add(element['text'])

        self.main_context['working_context'].delete_by_mid_list(memory_list)
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        text = query['text']

        # Trigger functions to update the current state of memory model.
        self.__trigger_function_call__(text)

        return self.get_current_memory_context()
    
class ReflexionRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()

        self.storage = kwargs['storage']
        self.insight = kwargs['insight']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        # No retrieval — mark empty for eval
        self._local.last_retrieved_ids = []
        self._local.last_retrieved_texts = []

        if self.storage.is_empty():
            if self.insight['global_insight']:
                memory_context = self.utilization({
                    'Insight': self.insight['global_insight']
                })
            else:
                return self.config.empty_memory
        else:
            if self.insight['global_insight']:
                memory_context = self.utilization({
                    'Insight': self.insight['global_insight'],
                    'Memory': self.storage.get_all_memory_in_order()
                })
            else:
                memory_context = self.utilization(self.storage.get_all_memory_in_order())
        
        return self.truncation(memory_context)


class MultimodalRAGRecall(BaseRecall):
    """MultiModal Memory Recall"""
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.multimodal_retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []
        
        ranking_ids = self.multimodal_retrieval(query)
        
        if len(ranking_ids) == 0:
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []
        
        # Collect ALL retrieved memories
        memories = []
        for mid in ranking_ids:
            mem = self.storage.get_memory_element_by_mid(int(mid))
            memories.append(mem)

        # Store all for eval
        self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(memories)

        # context_topk: only pass a subset to LLM
        context_topk = _get_context_topk(self.config)
        context_memories = memories[:context_topk] if context_topk else memories

        result = self.utilization(context_memories)
        return result


class NeuralGraphRecall(BaseRecall):
    """
    Neural Graph Memory Recall:
    Query-aware graph traversal for memory retrieval
    Combines embedding similarity with graph structure
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs['storage']  # GraphStorage
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # Graph traversal parameters
        self.max_depth = getattr(config, 'max_depth', 3)
        self.max_nodes = getattr(config, 'max_nodes', 10)
        self.traversal_threshold = getattr(config, 'traversal_threshold', 0.5)
        
        self.traversal_strategy = getattr(config, 'traversal_strategy', 'depth_first')
        self.initial_candidate_multiplier = getattr(config, 'initial_candidate_multiplier', 2)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.multimodal_retrieval])
    
    def __graph_traversal_depth_first__(self, query_embedding, start_node_ids, visited=None, depth=0):
        import torch
        
        if visited is None:
            visited = set()
        
        if depth >= self.max_depth:
            return []
        
        candidates = []
        
        for node_id in start_node_ids:
            if node_id in visited:
                continue
            
            visited.add(node_id)

            node_mid = self.storage.get_mid_by_node_id(node_id)
            if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                continue
            
            node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

            query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
            node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
            similarity = torch.matmul(query_norm, node_norm.T).item()

            if similarity >= self.traversal_threshold:
                candidates.append((node_id, similarity))

            neighbors = self.storage.get_neighbors(node_id)
            if neighbors:
                neighbor_candidates = self.__graph_traversal_depth_first__(
                    query_embedding, 
                    neighbors, 
                    visited, 
                    depth + 1
                )
                candidates.extend(neighbor_candidates)
        
        return candidates
    
    def __graph_traversal_breadth_first__(self, query_embedding, initial_node_ids):
        import torch

        expanded_nodes = set()
        for node_id in initial_node_ids:
            expanded_nodes.add(node_id)
            neighbors = self.storage.get_neighbors(node_id)
            for neighbor in neighbors:
                expanded_nodes.add(neighbor)

        final_similarities = []
        for node_id in expanded_nodes:
            try:
                node_mid = self.storage.get_mid_by_node_id(node_id)
                if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                    continue
                
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

                query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                similarity = torch.matmul(query_norm, node_norm.T).item()
                
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                final_similarities.append((node_id, boosted_similarity))
            except (KeyError, IndexError, Exception):
                continue

        final_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return final_similarities
    
    def __graph_traversal__(self, query_embedding, start_node_ids, visited=None, depth=0):
        if self.traversal_strategy == 'breadth_first':
            return self.__graph_traversal_breadth_first__(query_embedding, start_node_ids)
        else:
            return self.__graph_traversal_depth_first__(query_embedding, start_node_ids, visited, depth)
    
    def __deep_reasoning_expand__(self, query_embedding, query_concepts_set, query_entities,
                                  query_timestamp, seed_nodes):
        """Optional multi-hop reasoning to surface deeper nodes."""
        from collections import deque

        if not seed_nodes:
            self.last_reasoning_paths = {}
            self.last_reasoning_details = {}
            return {}

        results = {}
        queue = deque()
        for node_id in seed_nodes:
            queue.append((node_id, 0, [node_id], 1.0))

        while queue:
            current_node, depth, path, path_score = queue.popleft()
            if depth >= self.deep_reasoning_max_hops:
                continue

            neighbors = self.storage.get_neighbors(current_node)
            if not neighbors:
                continue

            neighbor_evaluations = []
            for neighbor in neighbors:
                if neighbor in path:
                    continue
                evaluation = self.__evaluate_node_alignment__(
                    query_embedding,
                    query_concepts_set,
                    query_entities,
                    query_timestamp,
                    neighbor
                )
                if evaluation is None:
                    continue
                combined_score, score_details, _ = evaluation
                neighbor_evaluations.append((neighbor, combined_score, score_details))

            if not neighbor_evaluations:
                continue

            neighbor_evaluations.sort(key=lambda item: item[1], reverse=True)
            trimmed = neighbor_evaluations[:self.deep_reasoning_branch_factor]

            for neighbor, combined_score, score_details in trimmed:
                edge_info = self.storage.get_edges_from(current_node).get(neighbor, {})
                edge_weight = float(edge_info.get('weight', 0.5))
                relation_type = edge_info.get('type', 'generic')

                adjusted_score = (combined_score + edge_weight) / 2.0
                cumulative_score = path_score * adjusted_score
                new_path = path + [neighbor]

                if cumulative_score >= self.deep_reasoning_threshold:
                    current_record = results.get(neighbor)
                    if current_record is None or cumulative_score > current_record['score']:
                        results[neighbor] = {
                            'score': cumulative_score,
                            'details': score_details,
                            'path': new_path,
                            'relation': relation_type,
                            'edge_weight': edge_weight
                        }

                if depth + 1 < self.deep_reasoning_max_hops:
                    queue.append((neighbor, depth + 1, new_path, cumulative_score))

        self.last_reasoning_paths = {node_id: info['path'] for node_id, info in results.items()}
        self.last_reasoning_details = results
        return results

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []

        initial_topk = 3
        if self.traversal_strategy == 'breadth_first':
            initial_topk = min(initial_topk * self.initial_candidate_multiplier, self.storage.get_element_number())
        
        ranking_ids = self.multimodal_retrieval(query, topk=min(initial_topk, self.storage.get_element_number()))
        
        if len(ranking_ids) == 0:
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []

        query_embedding = self.multimodal_retrieval.encoder(query, return_type='tensor')
        if self.multimodal_retrieval.config.mode == 'cosine':
            query_embedding = self.multimodal_retrieval.__normalize__(query_embedding)

        start_node_ids = []
        for mid in ranking_ids:
            try:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                start_node_ids.append(node_id)
            except (KeyError, IndexError):
                continue

        traversal_results = self.__graph_traversal__(query_embedding, start_node_ids)

        all_candidates = {}

        if self.traversal_strategy == 'breadth_first':
            for node_id, boosted_similarity in traversal_results:
                all_candidates[node_id] = boosted_similarity
        else:
            import torch
            for mid in ranking_ids:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                try:
                    node_mid = self.storage.get_mid_by_node_id(node_id)
                    if node_mid < self.multimodal_retrieval.tensorstore.size(0):
                        node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                        query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                        node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                        similarity = torch.matmul(query_norm, node_norm.T).item()
                        
                        node_degree = self.storage.get_node_degree(node_id)
                        degree_boost = 1 + (node_degree * 0.1)
                        all_candidates[node_id] = similarity * degree_boost
                    else:
                        all_candidates[node_id] = 1.0
                except Exception:
                    all_candidates[node_id] = 1.0

            for node_id, similarity in traversal_results:
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                if node_id not in all_candidates:
                    all_candidates[node_id] = boosted_similarity
                else:
                    all_candidates[node_id] = max(all_candidates[node_id], boosted_similarity)

        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        selected_node_ids = [node_id for node_id, _ in sorted_candidates[:self.max_nodes]]
        
        if not selected_node_ids:
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []

        # Collect ALL retrieved memories
        memories = []
        for node_id in selected_node_ids:
            mem = self.storage.get_memory_element_by_node_id(node_id)
            memories.append(mem)

        # Store all for eval
        self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(memories)

        # context_topk: only pass a subset to LLM
        context_topk = _get_context_topk(self.config)
        context_memories = memories[:context_topk] if context_topk else memories

        result = self.utilization(context_memories)
        return result


class AUGUSTUSRecall(BaseRecall):
    """
    AUGUSTUS Memory Recall:
    Implements hybrid retrieval with CoPe (Contextual-Personalized) search and embedding-based retrieval.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs.get('contextual_memory') or kwargs.get('storage')
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        self.concept_extractor = kwargs['concept_extractor']
        self.concept_retrieval = kwargs['concept_retrieval']
        
        if 'contextual_memory' not in kwargs and 'storage' in kwargs:
            self.storage = kwargs['storage']
        
        self.concept_retrieval.set_storage(self.storage)
        
        self.max_depth = getattr(config, 'max_depth', 3)
        self.max_nodes = getattr(config, 'max_nodes', 10)
        self.traversal_threshold = getattr(config, 'traversal_threshold', 0.5)
        
        self.embedding_weight = getattr(config, 'embedding_weight', 0.5)
        self.concept_weight = getattr(config, 'concept_weight', 0.5)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.multimodal_retrieval, self.concept_extractor, self.concept_retrieval])
    
    def cope_search(self, query):
        query_concepts = self.concept_extractor.extract(query)
        
        if not query_concepts:
            return []
        
        concept_counts = self.storage.get_nodes_by_concepts(query_concepts)
        
        matching_node_ids = []
        for node_id, overlap_count in concept_counts.items():
            if overlap_count >= self.concept_retrieval.min_concept_overlap:
                matching_node_ids.append(node_id)
        
        return matching_node_ids
    
    def __calculate_concept_score__(self, query_concepts_set, node_concepts):
        if not query_concepts_set or not node_concepts:
            return 0.0
        
        overlap = query_concepts_set & node_concepts
        return len(overlap) / max(len(query_concepts_set), len(node_concepts), 1)
    
    def __graph_traversal__(self, query_embedding, query_concepts, start_node_ids, visited=None, depth=0):
        import torch
        
        if visited is None:
            visited = set()
        
        if depth >= self.max_depth:
            return []
        
        candidates = []
        query_concepts_set = set(query_concepts) if query_concepts else set()
        
        for node_id in start_node_ids:
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                continue
            
            node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
            
            query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
            node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
            embedding_similarity = torch.matmul(query_norm, node_norm.T).item()
            
            node_concepts = self.storage.get_concepts_by_node(node_id)
            concept_score = self.__calculate_concept_score__(query_concepts_set, node_concepts)
            
            combined_score = (self.embedding_weight * embedding_similarity + 
                            self.concept_weight * concept_score)
            
            if combined_score >= self.traversal_threshold:
                candidates.append((node_id, combined_score))
            
            neighbors = self.storage.get_neighbors(node_id)
            if neighbors:
                neighbor_candidates = self.__graph_traversal__(
                    query_embedding,
                    query_concepts,
                    neighbors, 
                    visited, 
                    depth + 1
                )
                candidates.extend(neighbor_candidates)
        
        return candidates
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []
        
        ranking_ids = self.multimodal_retrieval(query, topk=min(3, self.storage.get_element_number()))
        
        query_concepts = self.concept_extractor.extract(query)
        
        cope_node_ids = []
        if query_concepts:
            cope_node_ids = self.cope_search(query)
        
        query_embedding = self.multimodal_retrieval.encoder(query, return_type='tensor')
        if self.multimodal_retrieval.config.mode == 'cosine':
            query_embedding = self.multimodal_retrieval.__normalize__(query_embedding)
        
        start_node_ids = []
        initial_scores = {}
        
        for mid in ranking_ids:
            try:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                start_node_ids.append(node_id)
                initial_scores[node_id] = 1.0
            except (KeyError, IndexError):
                continue
        
        concept_counts = self.storage.get_nodes_by_concepts(query_concepts) if query_concepts else {}
        for node_id in cope_node_ids:
            if node_id not in start_node_ids:
                start_node_ids.append(node_id)
            overlap_count = concept_counts.get(node_id, 0)
            initial_scores[node_id] = (overlap_count / len(query_concepts)) if query_concepts else 0.8
        
        if not start_node_ids:
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []
        
        traversal_results = []
        if start_node_ids:
            traversal_results = self.__graph_traversal__(
                query_embedding,
                query_concepts,
                start_node_ids
            )
        
        all_candidates = {}
        
        for node_id, score in initial_scores.items():
            all_candidates[node_id] = score
        
        for node_id, score in traversal_results:
            if node_id not in all_candidates:
                all_candidates[node_id] = 0.0
            all_candidates[node_id] = max(all_candidates[node_id], score)
        
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        selected_node_ids = [node_id for node_id, _ in sorted_candidates[:self.max_nodes]]
        
        if not selected_node_ids:
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []
        
        # Collect ALL retrieved memories
        memories = []
        for node_id in selected_node_ids:
            mem = self.storage.get_memory_element_by_node_id(node_id)
            memories.append(mem)
        
        # Store all for eval
        self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(memories)
        
        # context_topk: only pass a subset to LLM
        context_topk = _get_context_topk(self.config)
        context_memories = memories[:context_topk] if context_topk else memories
        
        result = self.utilization(context_memories)
        return result


class UniversalRAGRecall(BaseRecall):
    """
    UniversalRAG Recall
    Supports 5 types: 'no', 'document', 'image', 'audio', 'video'
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._local = threading.local()
        self.storage = kwargs['storage']
        self.routing = kwargs['routing']
        self.retrieval = kwargs['retrieval']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.routing, self.retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []

        modality = self.routing.route(query)

        if modality == 'no':
            self._local.last_retrieved_ids = []
            self._local.last_retrieved_texts = []
            return []

        retrieved_ids, scores = self.retrieval.retrieve(
            query, modality,
            top_k=getattr(self.config, 'top_k', 5)
        )

        if len(retrieved_ids) == 0 and modality in ('image', 'audio', 'video'):
            retrieved_ids, scores = self.retrieval.retrieve(
                query, 'document',
                top_k=getattr(self.config, 'top_k', 5)
            )

        # Collect ALL retrieved memories
        memories = []
        for mid in retrieved_ids:
            mem = self.storage.get_memory_element_by_mid(mid)
            memories.append(mem)

        # Store all for eval
        self._local.last_retrieved_ids, self._local.last_retrieved_texts = _extract_eval_data(memories)

        # context_topk: only pass a subset to LLM
        context_topk = _get_context_topk(self.config)
        context_memories = memories[:context_topk] if context_topk else memories

        result = self.utilization(context_memories)
        return result