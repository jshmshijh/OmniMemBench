from abc import ABC, abstractmethod
from transformers import AutoTokenizer

class BaseTruncation(ABC):
    """
    Helps to formulate memory contexts under the limitations of token number by certain LLMs.
    """
    def __init__(self, config) -> None:
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LMTruncation(BaseTruncation):
    def __init__(self, config):
        super().__init__(config)

        if self.config.mode == 'token':
            print(f"[LMTruncation] Loading tokenizer from: {self.config.path} ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)
            print(f"[LMTruncation] Tokenizer loaded.", flush=True)

    def truncate_by_word(self, text):
        words = text.split(' ')
        if len(words) > self.config.number:
            words = words[-self.config.number:]
        return ' '.join(words)

    def truncate_by_token(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.config.number:
            tokens = tokens[-self.config.number:]
        truncated_text = self.tokenizer.convert_tokens_to_string(tokens)
        return truncated_text

    def get_piece_number(self, text):
        if self.config.mode == 'word':
            return len(text.split(' '))
        elif self.config.mode == 'token':
            return len(self.tokenizer.tokenize(text))
        else:
            raise "Truncation mode error."

    def check_truncation_needed(self, text):
        return self.get_piece_number(text) > self.config.number

    def __call__(self, text):
        if self.config.mode == 'word':
            return self.truncate_by_word(text)
        elif self.config.mode == 'token':
            return self.truncate_by_token(text)
        else:
            raise "Truncation mode error."


class MMLMTruncation(BaseTruncation):
    """
    MultiModal Language Model Truncation:
    Truncates multimodal memory based on token budget (text tokens + image tokens).
    Formula: N_text + (N_image × T_img) ≤ L_max
    
    Truncates from the oldest memories (keeps the newest ones).
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize tokenizer for text token counting
        if self.config.mode == 'token':
            print(f"[MMLMTruncation] Loading tokenizer from: {self.config.path} ...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)
            print(f"[MMLMTruncation] Tokenizer loaded.", flush=True)
        else:
            self.tokenizer = None
        
        # Tokens per image (T_img)
        # Set to 0: media tokens are not counted, truncation is controlled by text token budget only
        self.tokens_per_image = getattr(self.config, 'tokens_per_image', 0)
        self.tokens_per_audio = getattr(self.config, 'tokens_per_audio', 0)
        self.tokens_per_video = getattr(self.config, 'tokens_per_video', 0)
        
        # Maximum total tokens (L_max)
        self.max_tokens = self.config.number
    
    def reset(self):
        pass
    
    def _count_text_tokens(self, text):
        """Count tokens in text."""
        if self.config.mode == 'token' and self.tokenizer:
            return len(self.tokenizer.tokenize(text))
        elif self.config.mode == 'word':
            return len(text.split(' '))
        else:
            raise "Truncation mode error."
    
    def _count_image_tokens(self, memory_item):
        """Count tokens for images in a memory item."""
        image = memory_item.get('image')
        if image and image is not None:
            if isinstance(image, dict):
                return self.tokens_per_image
            elif isinstance(image, str) and image.strip():
                return self.tokens_per_image
        return 0

    def _count_audio_tokens(self, memory_item):
        """Count tokens for audio in a memory item."""
        audio = memory_item.get('audio')
        if audio and isinstance(audio, dict):
            return self.tokens_per_audio
        return 0

    def _count_video_tokens(self, memory_item):
        """Count tokens for video in a memory item."""
        video = memory_item.get('video')
        if video and isinstance(video, dict):
            return self.tokens_per_video
        return 0

    def _count_memory_tokens(self, memory_item):
        """Count total tokens for a memory item (text + image + audio + video)."""
        text = memory_item.get('text', '')
        text_tokens  = self._count_text_tokens(text)
        image_tokens = self._count_image_tokens(memory_item)
        audio_tokens = self._count_audio_tokens(memory_item)
        video_tokens = self._count_video_tokens(memory_item)
        return text_tokens + image_tokens + audio_tokens + video_tokens
    
    def __call__(self, memory_list):
        """
        Truncate multimodal memory list based on token budget AND image count limit.
        
        Logic:
        - If memory has image: check both image count limit AND token budget
        - If memory has no image: only check token budget
        - This allows adding text-only memories even after image limit is reached
        
        Args:
            memory_list: list of dict, each dict is a memory item with:
                - 'text': str
                - 'image': dict or None
                - 'timestamp': int/float
                - 'dialogue_id': int
                - etc.
        
        Returns:
            list: Truncated memory list (keeping newest memories)
        """
        if not memory_list:
            return []
        
        # Maximum number of media items allowed (default: 999, effectively unlimited)
        max_media = getattr(self.config, 'max_media', getattr(self.config, 'max_images', 999))

        # Process from newest to oldest (reverse order)
        result = []
        total_tokens = 0
        total_media = 0

        # Iterate from newest (last) to oldest (first)
        for mem in reversed(memory_list):
            mem_tokens = self._count_memory_tokens(mem)
            has_media = (self._count_image_tokens(mem) + self._count_audio_tokens(mem) + self._count_video_tokens(mem)) > 0

            # Check if memory can be added
            can_add = False

            if has_media:
                # Memory has media: check both media count limit AND token budget
                if (total_media < max_media and
                        total_tokens + mem_tokens <= self.max_tokens):
                    can_add = True
            else:
                # Memory has no media: only check token budget
                if total_tokens + mem_tokens <= self.max_tokens:
                    can_add = True

            if can_add:
                # Insert at beginning to maintain chronological order (oldest first)
                result.insert(0, mem)
                total_tokens += mem_tokens
                if has_media:
                    total_media += 1
            else:
                # Cannot add this memory
                if not has_media and total_tokens + mem_tokens > self.max_tokens:
                    # Token budget exceeded, stop adding more (older) memories
                    break
                # Otherwise, continue to next memory (might be text-only that can fit)
                continue

        return result