"""
Music Embeddings Extraction Package

This package provides tools for extracting embeddings from audio files
using various pre-trained models like OpenL3 and AudioCLIP.

Author: Sergie Code
Purpose: AI tools for musicians and audio analysis
"""

__version__ = "1.0.0"
__author__ = "Sergie Code"
__email__ = "sergiecode@example.com"

from .embeddings import AudioEmbeddingExtractor
from .utils import load_audio, preprocess_audio, save_embeddings, load_embeddings

__all__ = [
    'AudioEmbeddingExtractor',
    'load_audio',
    'preprocess_audio', 
    'save_embeddings',
    'load_embeddings'
]
