from omnimembench.engine.utils.Storage import BaseStorage
from omnimembench.engine.function.MultiModalEncoder import GMEEncoder, OmniEncoder
import torch

class UniversalRAGStorage(BaseStorage):
    """
    UniversalRAG Storage
    Supports text, image, audio, and video indexing for multi-modal retrieval.
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.memory_list = []
        self.counter = 0
        
        self._init_encoder(config)
        
        self.document_index = {}      
        self.image_index = {}
        self.audio_index = {}
        self.video_index = {}
    
    def _init_encoder(self, config):
        """Initialize encoder (supports GMEEncoder and OmniEncoder)"""
        encoder_config = getattr(config, 'encoder', None)
        if encoder_config:
            encoder_method = getattr(encoder_config, 'method', 'GMEEncoder')
            if encoder_method == 'OmniEncoder':
                self.encoder = OmniEncoder(encoder_config)
            else:
                self.encoder = GMEEncoder(encoder_config)
        else:
            DEFAULT_GME_ENCODER = None  # loaded from config at runtime
            self.encoder = GMEEncoder(DEFAULT_GME_ENCODER)
    
    def reset(self):
        self.memory_list = []
        self.counter = 0
        self.document_index = {}
        self.image_index = {}
        self.audio_index = {}
        self.video_index = {}
    
    def is_empty(self):
        return self.counter == 0
    
    def display(self):
        memory_display_items = []
        for m in self.memory_list:
            memory_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in m.items()]))
        if len(memory_display_items) == 0:
            return 'None'
        return '\n'.join(['[Memory Entity %d]\n%s' % (index, m) for index, m in enumerate(memory_display_items)])
    
    def add(self, obj):
        """
        Add observations and extract features.
        - document_index: Extract text features (for text retrieval)
        - image_index: Extract image features (for image retrieval)
        - audio_index: Extract audio features (for audio retrieval, requires OmniEncoder)
        - video_index: Extract video features (for video retrieval, requires OmniEncoder)
        """
        mid = self.counter
        
        obj['counter_id'] = mid
        self.memory_list.append(obj)
        
        # Document index: Extract text features (even if the observation includes an image)
        text = obj.get('text', '') if isinstance(obj, dict) else str(obj) if isinstance(obj, str) else ''
        if text or not isinstance(obj, dict) or 'image' not in obj:
            doc_embedding = self.encoder.encode_text(text, return_type='tensor')
            self.document_index[mid] = doc_embedding
        elif isinstance(obj, dict) and 'image' in obj:
            doc_embedding = self.encoder.encode_text(' ', return_type='tensor')
            self.document_index[mid] = doc_embedding
        
        # Image index: If an image is included, extract image features.
        if isinstance(obj, dict) and 'image' in obj and obj['image']:
            image_path = obj['image'].get('path') if isinstance(obj['image'], dict) else obj['image']
            if image_path:
                img_embedding = self.encoder.encode_image(image_path, return_type='tensor')
                self.image_index[mid] = img_embedding
        
        # Audio index: If audio is included, extract audio features (requires OmniEncoder).
        if isinstance(obj, dict) and 'audio' in obj and obj['audio']:
            if isinstance(self.encoder, OmniEncoder):
                try:
                    audio_obj = {'audio': obj['audio']}
                    audio_embedding = self.encoder(audio_obj, return_type='tensor')
                    self.audio_index[mid] = audio_embedding
                except Exception as e:
                    print(f"Warning: Failed to encode audio for mid {mid}: {e}. Skipping audio index.")
            else:
                # Non-OmniEncoder: use caption text as fallback embedding
                caption = obj['audio'].get('caption', '') if isinstance(obj['audio'], dict) else ''
                if caption:
                    audio_embedding = self.encoder.encode_text(caption, return_type='tensor')
                    self.audio_index[mid] = audio_embedding

        # Video index: If video is included, extract video features (requires OmniEncoder).
        if isinstance(obj, dict) and 'video' in obj and obj['video']:
            if isinstance(self.encoder, OmniEncoder):
                try:
                    video_obj = {'video': obj['video']}
                    video_embedding = self.encoder(video_obj, return_type='tensor')
                    self.video_index[mid] = video_embedding
                except Exception as e:
                    print(f"Warning: Failed to encode video for mid {mid}: {e}. Skipping video index.")
            else:
                # Non-OmniEncoder: use caption text as fallback embedding
                caption = obj['video'].get('caption', '') if isinstance(obj['video'], dict) else ''
                if caption:
                    video_embedding = self.encoder.encode_text(caption, return_type='tensor')
                    self.video_index[mid] = video_embedding
        
        self.counter += 1
    
    def get_memory_element_by_mid(self, mid):
        return self.memory_list[mid]
    
    def get_memory_text_by_mid(self, mid):
        obj = self.memory_list[mid]
        if 'text' in obj:
            text = obj['text']
            return text if isinstance(text, str) else ' '.join(text)
        return ''
    
    def get_memory_image_by_mid(self, mid):
        """Get image information of a specified memory"""
        obj = self.memory_list[mid]
        return obj.get('image', None)

    def get_memory_audio_by_mid(self, mid):
        """Get audio information of a specified memory"""
        obj = self.memory_list[mid]
        return obj.get('audio', None)

    def get_memory_video_by_mid(self, mid):
        """Get video information of a specified memory"""
        obj = self.memory_list[mid]
        return obj.get('video', None)
    
    def _retrieve_from_index(self, query_embedding, features, top_k):
        """
        Core retrieval logic: compute similarity between query embedding and feature index.

        Args:
            query_embedding: Query embedding tensor
            features: dict {mid: embedding_tensor}
            top_k: Return top-k results

        Returns:
            (similarities_dict): {mid: score}
        """
        if not features:
            return {}

        query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        similarities = {}
        for mid, feat in features.items():
            feat_norm = feat / feat.norm(dim=-1, keepdim=True)
            sim = torch.matmul(query_norm, feat_norm.T).item()
            similarities[mid] = sim
        return similarities

    def _supplement_from_document(self, similarities, query, top_k):
        """
        Supplement retrieval results from document_index when primary index results are insufficient.
        """
        if len(similarities) >= top_k or not self.document_index:
            return similarities

        text = query.get('text', '') if isinstance(query, dict) else str(query)
        text_query_embedding = self.encoder.encode_text(text, return_type='tensor')
        text_query_norm = text_query_embedding / text_query_embedding.norm(dim=-1, keepdim=True)

        for mid, feat in self.document_index.items():
            feat_norm = feat / feat.norm(dim=-1, keepdim=True)
            text_sim = torch.matmul(text_query_norm, feat_norm.T).item()

            if mid not in similarities:
                similarities[mid] = text_sim
            else:
                similarities[mid] = (similarities[mid] + text_sim) / 2.0
        return similarities

    def retrieve_by_query(self, query, modality, top_k=5):
        """
        Retrieve based on query and modality.

        Args:
            query: Query object (dict or str)
            modality: 'no', 'document', 'image', 'audio', or 'video'
            top_k: Return top-k results

        Returns:
            (retrieved_mids, scores): List of retrieved mids and scores
        """
        if modality == 'no':
            return [], []
        
        # Encode query based on modality
        if modality == 'image':
            if isinstance(query, dict) and 'image' in query and query['image']:
                image_path = query['image'].get('path') if isinstance(query['image'], dict) else query['image']
                if image_path:
                    query_embedding = self.encoder.encode_image(image_path, return_type='tensor')
                else:
                    text = query.get('text', '') if isinstance(query, dict) else str(query)
                    query_embedding = self.encoder.encode_text(text, return_type='tensor')
            else:
                text = query.get('text', '') if isinstance(query, dict) else str(query)
                query_embedding = self.encoder.encode_text(text, return_type='tensor')
            features = self.image_index

        elif modality == 'audio':
            # Audio retrieval: use OmniEncoder for audio query, or fallback to text
            if isinstance(self.encoder, OmniEncoder) and isinstance(query, dict) and 'audio' in query and query['audio']:
                try:
                    audio_query = {'audio': query['audio']}
                    query_embedding = self.encoder(audio_query, return_type='tensor')
                except Exception:
                    text = query.get('text', '') if isinstance(query, dict) else str(query)
                    query_embedding = self.encoder.encode_text(text, return_type='tensor')
            else:
                text = query.get('text', '') if isinstance(query, dict) else str(query)
                query_embedding = self.encoder.encode_text(text, return_type='tensor')
            features = self.audio_index

        elif modality == 'video':
            # Video retrieval: use OmniEncoder for video query, or fallback to text
            if isinstance(self.encoder, OmniEncoder) and isinstance(query, dict) and 'video' in query and query['video']:
                try:
                    video_query = {'video': query['video']}
                    query_embedding = self.encoder(video_query, return_type='tensor')
                except Exception:
                    text = query.get('text', '') if isinstance(query, dict) else str(query)
                    query_embedding = self.encoder.encode_text(text, return_type='tensor')
            else:
                text = query.get('text', '') if isinstance(query, dict) else str(query)
                query_embedding = self.encoder.encode_text(text, return_type='tensor')
            features = self.video_index

        else:  # document
            text = query.get('text', '') if isinstance(query, dict) else str(query)
            query_embedding = self.encoder.encode_text(text, return_type='tensor')
            features = self.document_index
        
        # Core retrieval
        similarities = self._retrieve_from_index(query_embedding, features, top_k)

        # Supplement from document_index if primary modality results are insufficient
        if modality in ('image', 'audio', 'video'):
            similarities = self._supplement_from_document(similarities, query, top_k)
        
        # Sort and return top-k
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved_mids = [mid for mid, _ in sorted_items]
        scores = [score for _, score in sorted_items]
        
        return retrieved_mids, scores

