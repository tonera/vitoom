"""Text tokenization for Qwen3-TTS."""
from transformers import AutoTokenizer
import torch
from typing import List, Optional


class TTSTokenizer:
    """Wrapper for Qwen2 tokenizer used in Qwen3-TTS.
    
    Qwen3-TTS uses Qwen2's tokenizer for text input processing.
    Vocab size: 151936 tokens.
    """
    
    def __init__(self, model_path: str):
        """Initialize tokenizer.
        
        Args:
            model_path: Path to model directory containing tokenizer files
        """
        # Qwen3-TTS uses Qwen2's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        
        # Cache special token IDs
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id
        
        print(f"Loaded tokenizer: vocab_size={len(self.tokenizer)}")
        print(f"  BOS: {self.bos_token_id}, EOS: {self.eos_token_id}, PAD: {self.pad_token_id}")
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add default special tokens
            add_bos: Explicitly add BOS token at start
            add_eos: Explicitly add EOS token at end
            
        Returns:
            List of token IDs
        """
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens
        )
        
        # Add explicit BOS/EOS if requested
        if add_bos and token_ids[0] != self.bos_token_id:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos and token_ids[-1] != self.eos_token_id:
            token_ids = token_ids + [self.eos_token_id]
        
        return token_ids
    
    def encode_batch(
        self, 
        texts: List[str],
        padding: bool = False,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """Encode batch of texts.
        
        Args:
            texts: List of text strings
            padding: Whether to pad to same length
            max_length: Maximum length (for padding/truncation)
            
        Returns:
            List of token ID lists (or padded tensor if padding=True)
        """
        if padding:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length,
                return_tensors="pt",
            )
            return encoded["input_ids"].tolist()
        else:
            return [self.encode(text) for text in texts]
    
    def decode(
        self, 
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(
            token_ids, 
            skip_special_tokens=skip_special_tokens
        )
    
    def decode_batch(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode batch of token IDs.
        
        Args:
            token_ids_list: List of token ID lists
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            List of decoded text strings
        """
        return [
            self.decode(token_ids, skip_special_tokens)
            for token_ids in token_ids_list
        ]
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
