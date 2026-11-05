from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import os
import warnings
import numpy as np
import tempfile

if hasattr(torch.cuda, '_queued_calls'):
    torch.cuda._queued_calls.clear()
if torch.cuda.is_available():
    torch.cuda.init()
    
class BaseMultiModalEncoder(ABC):
    """
    Encoder for multimodal data (text + image).
    """
    def __init__(self, config):
        self.config = config
        _no_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') in ('-1', 'NoDevFiles')
        self.device = torch.device('cpu') if (_no_cuda or not torch.cuda.is_available()) else torch.device('cuda:0')
    
    def reset(self):
        pass
    
    @abstractmethod
    def encode_text(self, text, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_image(self, image_path_or_url, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        pass


class CLIPEncoder(BaseMultiModalEncoder):
    """
    CLIP-based multimodal encoder for text and images.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'path', 'openai/clip-vit-base-patch32')
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        print(f"CLIP model loaded successfully on {self.device}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            image_path_or_url = image_path_or_url
        else:
            # Only relative paths are prefixed.
            image_path_or_url = "" + image_path_or_url # [Replace with your default absolute path]
        print("image_path_or_url: ", image_path_or_url)
        try:
            if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
                # Load from URL
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path_or_url}: {e}")
            return Image.new('RGB', (224, 224), color='white')
    
    def encode_text(self, text, return_type='numpy'):
        """Encode text into embeddings."""
        if not text or text.strip() == '':
            text = " "
        
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return text_features.cpu().numpy()
        elif return_type == 'tensor':
            return text_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_image(self, image_path_or_url, return_type='numpy'):
        """Encode image into embeddings."""
        image = self._load_image(image_path_or_url)
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return image_features.cpu().numpy()
        elif return_type == 'tensor':
            return image_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        """
        Encode multimodal data (text and/or image).
        If both are provided, average the embeddings.
        """
        embeddings = []
        
        if text is not None and text.strip() != '':
            text_emb = self.encode_text(text, return_type='tensor')
            embeddings.append(text_emb)
        
        if image is not None:
            image_emb = self.encode_image(image.get('path'), return_type='tensor')
            embeddings.append(image_emb)
        
        if not embeddings:
            # If both are empty, encode empty text
            return self.encode_text(" ", return_type=return_type)
        
        # Average the embeddings if both modalities are present
        if len(embeddings) > 1:
            combined = torch.mean(torch.stack(embeddings), dim=0)
        else:
            combined = embeddings[0]
        
        # Normalize
        combined = combined / combined.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return combined.cpu().numpy()
        elif return_type == 'tensor':
            return combined
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def __call__(self, obj, return_type='numpy'):
        """
        Main entry point. obj can be:
        - str: treated as text
        - dict with 'text' and/or 'image' keys
        """
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")


class OmniEncoder(BaseMultiModalEncoder):
    """
    OmniEncoder: Omni embedding for all modalities (text/image/audio/video).
    Uses nvidia/omni-embed-nemotron-3b as the underlying model.

    config.emb_mode controls the embedding strategy:
      - 'caption' (default): concatenate all captions → text-only embedding
      - 'raw': pass actual media (HTTP URLs) → full omni embedding
    """

    def __init__(self, config):
        super().__init__(config)
        from qwen_omni_utils import process_mm_info
        self.process_mm_info = process_mm_info

        if self.device.type == 'cuda':
            torch.cuda.init()

        model_path = getattr(config, 'path', 'nvidia/omni-embed-nemotron-3b')
        print(f"Loading OmniEncoder (Nemotron backbone): {model_path}")
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        except (ImportError, ValueError):
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.emb_mode = getattr(config, 'emb_mode', 'caption')
        print(f"OmniEncoder loaded successfully on {self.device}, emb_mode={self.emb_mode}")

    def _get_embedding(self, messages):
        """Run Nemotron forward pass and return normalized embedding tensor."""
        texts = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        audio, images, videos = self.process_mm_info(messages, use_audio_in_video=False)
        batch_dict = self.processor(
            text=texts,
            images=images,
            videos=videos,
            audio=audio,
            return_tensors="pt",
            text_kwargs={"truncation": True, "padding": True, "max_length": 204800},
            videos_kwargs={"min_pixels": 32 * 14 * 14, "max_pixels": 64 * 28 * 28, "use_audio_in_video": False},
            audio_kwargs={"max_length": 2048000},
        )
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            last_hidden_states = self.model(**batch_dict, output_hidden_states=True).hidden_states[-1]
        attention_mask = batch_dict["attention_mask"]
        last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return F.normalize(embedding, dim=-1)

    def _to_output(self, emb, return_type):
        if return_type == 'numpy':
            return emb.cpu().float().numpy()
        elif return_type == 'tensor':
            return emb
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")

    def encode_text(self, text, return_type='numpy'):
        if not text or text.strip() == '':
            text = ' '
        messages = [{"role": "user", "content": [{"type": "text", "text": f"passage: {text}"}]}]
        return self._to_output(self._get_embedding(messages), return_type)

    def encode_image(self, image_path_or_url, return_type='numpy'):
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path_or_url}]}]
        return self._to_output(self._get_embedding(messages), return_type)

    def _caption_emb(self, obj, return_type='numpy'):
        """Concatenate all available captions → text-only embedding."""
        parts = []
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            if obj.get('text'):
                parts.append(obj['text'])
            if obj.get('image') and isinstance(obj['image'], dict) and obj['image'].get('caption'):
                parts.append(obj['image']['caption'])
            if obj.get('audio') and isinstance(obj['audio'], dict) and obj['audio'].get('caption'):
                parts.append(obj['audio']['caption'])
            if obj.get('video') and isinstance(obj['video'], dict) and obj['video'].get('caption'):
                parts.append(obj['video']['caption'])
        return self.encode_text(' '.join(parts) or ' ', return_type)

    def _raw_emb(self, obj, return_type='numpy'):
        """Full omni embedding: pass actual media HTTP URLs to Nemotron."""
        content = []
        if isinstance(obj, str):
            content.append({"type": "text", "text": f"passage: {obj}"})
        elif isinstance(obj, dict):
            if obj.get('text'):
                content.append({"type": "text", "text": f"passage: {obj['text']}"})
            if obj.get('image') and isinstance(obj['image'], dict) and obj['image'].get('path'):
                content.append({"type": "image", "image": obj['image']['path']})
            if obj.get('audio') and isinstance(obj['audio'], dict) and obj['audio'].get('url'):
                content.append({"type": "audio", "audio": obj['audio']['url']})
            if obj.get('video') and isinstance(obj['video'], dict) and obj['video'].get('url'):
                # Pre-extract frames with decord to avoid std::bad_alloc from process_mm_info
                video_url = obj['video']['url']
                try:
                    from decord import VideoReader, cpu
                    tmp_file = None
                    if video_url.startswith('http://') or video_url.startswith('https://'):
                        resp = requests.get(video_url, timeout=30)
                        tmp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                        tmp_file.write(resp.content)
                        tmp_file.flush()
                        video_path = tmp_file.name
                    else:
                        video_path = video_url
                    vr = VideoReader(video_path, ctx=cpu(0))
                    n = min(32, len(vr))
                    indices = np.linspace(0, len(vr) - 1, n, dtype=int)
                    frames = [Image.fromarray(vr[int(i)].asnumpy()).resize((448, 448)) for i in indices]
                    content.append({"type": "video", "video": frames})
                    if tmp_file:
                        os.unlink(tmp_file.name)
                except Exception as e:
                    print(f"[OmniEncoder] Video frame extraction failed: {e}, falling back to caption")
                    if obj['video'].get('caption'):
                        content.append({"type": "text", "text": f"passage: {obj['video']['caption']}"})
        if not content:
            content = [{"type": "text", "text": "passage: "}]
        messages = [{"role": "user", "content": content}]
        return self._to_output(self._get_embedding(messages), return_type)

    def encode_multimodal(self, obj=None, image=None, return_type='numpy'):
        if self.emb_mode == 'raw':
            return self._raw_emb(obj, return_type)
        return self._caption_emb(obj, return_type)

    def __call__(self, obj, return_type='numpy'):
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        return self.encode_multimodal(obj, return_type=return_type)


class GMEEncoder(BaseMultiModalEncoder):
    """
    GME (General Multimodal Embedding) Qwen2-VL-based encoder for text and images.
    Supports unified multimodal representations for Any2Any Search.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'path', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')
        print(f"Loading GME model: {model_name}")
        
        # Load GME model with trust_remote_code=True
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.eval()  # Set to evaluation mode
        print(f"GME model loaded successfully on {self.device}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        # If already absolute path or URL, use directly
        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            final_path = image_path_or_url
        else:
            # Only add prefix for relative paths
            final_path = "" + image_path_or_url # [Replace with your default absolute path]
        
        try:
            if final_path.startswith('http://') or final_path.startswith('https://'):
                # Load from URL
                response = requests.get(final_path, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                if not os.path.exists(final_path):
                    raise FileNotFoundError(f"Image file not found: {final_path}")
                image = Image.open(final_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path_or_url} (tried {final_path}): {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='white')
    
    def encode_text(self, text, return_type='numpy'):
        """Encode text into embeddings."""
        if not text or text.strip() == '':
            text = " "
        
        with torch.no_grad():
            text_emb = self.model.get_text_embeddings(texts=[text])
            # Normalize
            text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return text_emb.cpu().numpy()
        elif return_type == 'tensor':
            return text_emb
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_image(self, image_path_or_url, return_type='numpy'):
        """Encode image into embeddings."""
        image = self._load_image(image_path_or_url)
        
        with torch.no_grad():
            # GME API: get_image_embeddings expects images as list of PIL Image
            image_emb = self.model.get_image_embeddings(images=[image])
            # Normalize
            image_emb = image_emb / torch.norm(image_emb, dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return image_emb.cpu().numpy()
        elif return_type == 'tensor':
            return image_emb
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        """
        Encode multimodal data (text and/or image).
        GME supports single-modal (text/image) and fused-modal embeddings.
        If both are provided, we use fused-modal embedding for better representation.
        """
        # Determine which encoding method to use
        has_text = text is not None and text.strip() != ''
        has_image = image is not None
        
        if has_text and has_image:
            # Fused-modal embedding: best for multimodal representation
            image_path = image.get('path')
            loaded_image = self._load_image(image_path)
            with torch.no_grad():
                fused_emb = self.model.get_fused_embeddings(texts=[text], images=[loaded_image])
                fused_emb = fused_emb / torch.norm(fused_emb, dim=-1, keepdim=True)
            
            if return_type == 'numpy':
                return fused_emb.cpu().numpy()
            elif return_type == 'tensor':
                return fused_emb
            else:
                raise ValueError(f"Unrecognized return type: {return_type}")
        
        elif has_text:
            # Single-modal text embedding
            return self.encode_text(text, return_type)
        
        elif has_image:
            # Single-modal image embedding
            return self.encode_image(image.get('path'), return_type)
        
        else:
            # If both are empty, encode empty text
            return self.encode_text(" ", return_type)
    
    def __call__(self, obj, return_type='numpy'):
        """
        Main entry point. obj can be:
        - str: treated as text
        - dict with 'text' and/or 'image' keys
        """
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")