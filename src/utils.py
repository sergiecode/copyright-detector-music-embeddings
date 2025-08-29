"""
Audio Utilities Module

This module provides helper functions for audio processing, loading, 
preprocessing, and embedding management.

Author: Sergie Code
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pickle
import json
from typing import Tuple, Optional, Dict, Any
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


def load_audio(file_path: str, 
               target_sr: Optional[int] = None,
               mono: bool = True,
               offset: float = 0.0,
               duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.
    
    Args:
        file_path (str): Path to the audio file
        target_sr (int, optional): Target sample rate. If None, uses original sample rate
        mono (bool): Convert to mono if True
        offset (float): Start reading after this time (in seconds)
        duration (float, optional): Only load up to this much audio (in seconds)
    
    Returns:
        Tuple[np.ndarray, int]: Audio time series and sample rate
    
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        Exception: If there's an error loading the audio
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        audio, sr = librosa.load(
            file_path,
            sr=target_sr,
            mono=mono,
            offset=offset,
            duration=duration
        )
        
        print(f"✓ Loaded audio: {file_path}")
        print(f"  - Shape: {audio.shape}")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - Duration: {len(audio) / sr:.2f} seconds")
        
        return audio, sr
        
    except Exception as e:
        raise Exception(f"Error loading audio file {file_path}: {str(e)}")


def preprocess_audio(audio: np.ndarray, 
                    target_sr: int = 22050,
                    normalize: bool = True,
                    trim_silence: bool = True,
                    max_duration: Optional[float] = None) -> np.ndarray:
    """
    Preprocess audio data for embedding extraction.
    
    Args:
        audio (np.ndarray): Input audio array
        target_sr (int): Target sample rate
        normalize (bool): Whether to normalize the audio
        trim_silence (bool): Whether to trim leading/trailing silence
        max_duration (float, optional): Maximum duration in seconds
    
    Returns:
        np.ndarray: Preprocessed audio
    """
    processed_audio = audio.copy()
    
    # Trim silence
    if trim_silence:
        processed_audio, _ = librosa.effects.trim(processed_audio, top_db=20)
        print("✓ Trimmed silence")
    
    # Limit duration if specified
    if max_duration:
        max_samples = int(max_duration * target_sr)
        if len(processed_audio) > max_samples:
            processed_audio = processed_audio[:max_samples]
            print(f"✓ Truncated to {max_duration} seconds")
    
    # Normalize audio
    if normalize:
        if np.max(np.abs(processed_audio)) > 0:
            processed_audio = processed_audio / np.max(np.abs(processed_audio))
            print("✓ Normalized audio")
    
    return processed_audio


def save_embeddings(embeddings: np.ndarray, 
                   file_path: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   format: str = 'pickle') -> None:
    """
    Save embeddings to disk with optional metadata.
    
    Args:
        embeddings (np.ndarray): Embedding vectors to save
        file_path (str): Output file path
        metadata (dict, optional): Additional metadata to save
        format (str): Save format ('pickle', 'npy', or 'npz')
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'pickle':
        data = {
            'embeddings': embeddings,
            'metadata': metadata or {}
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    elif format == 'npy':
        np.save(file_path, embeddings)
        if metadata:
            metadata_path = file_path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    elif format == 'npz':
        if metadata:
            np.savez(file_path, embeddings=embeddings, metadata=np.array([metadata]))
        else:
            np.savez(file_path, embeddings=embeddings)
    
    print(f"✓ Saved embeddings to: {file_path}")


def load_embeddings(file_path: str, format: str = 'auto') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load embeddings from disk.
    
    Args:
        file_path (str): Path to the embeddings file
        format (str): File format ('auto', 'pickle', 'npy', or 'npz')
    
    Returns:
        Tuple[np.ndarray, Dict]: Embeddings and metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")
    
    # Auto-detect format
    if format == 'auto':
        if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            format = 'pickle'
        elif file_path.endswith('.npy'):
            format = 'npy'
        elif file_path.endswith('.npz'):
            format = 'npz'
        else:
            raise ValueError(f"Cannot auto-detect format for file: {file_path}")
    
    metadata = {}
    
    if format == 'pickle':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            embeddings = data['embeddings']
            metadata = data.get('metadata', {})
            
    elif format == 'npy':
        embeddings = np.load(file_path)
        metadata_path = file_path.replace('.npy', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
    elif format == 'npz':
        data = np.load(file_path, allow_pickle=True)
        embeddings = data['embeddings']
        if 'metadata' in data:
            metadata = data['metadata'].item()
    
    print(f"✓ Loaded embeddings from: {file_path}")
    print(f"  - Shape: {embeddings.shape}")
    
    return embeddings, metadata


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about an audio file.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        Dict[str, Any]: Audio file information
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Get basic info using soundfile
        info = sf.info(file_path)
        
        # Load a small sample to get additional info
        audio, sr = librosa.load(file_path, duration=30.0)  # First 30 seconds
        
        # Calculate additional metrics
        rms_energy = np.sqrt(np.mean(audio**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
        
        audio_info = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'duration_seconds': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'rms_energy': float(rms_energy),
            'zero_crossing_rate': float(zero_crossing_rate),
            'spectral_centroid_hz': float(spectral_centroid),
        }
        
        return audio_info
        
    except Exception as e:
        raise Exception(f"Error getting audio info for {file_path}: {str(e)}")


def create_audio_segments(audio: np.ndarray, 
                         sr: int,
                         segment_duration: float = 10.0,
                         overlap: float = 0.5) -> list:
    """
    Split audio into overlapping segments for batch processing.
    
    Args:
        audio (np.ndarray): Input audio array
        sr (int): Sample rate
        segment_duration (float): Duration of each segment in seconds
        overlap (float): Overlap between segments (0.0 to 1.0)
    
    Returns:
        list: List of audio segments
    """
    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    # Add the last segment if there's remaining audio
    if start < len(audio):
        remaining = audio[start:]
        # Pad with zeros if too short
        if len(remaining) < segment_samples:
            padded = np.zeros(segment_samples)
            padded[:len(remaining)] = remaining
            segments.append(padded)
        else:
            segments.append(remaining)
    
    print(f"✓ Created {len(segments)} audio segments")
    return segments


def validate_audio_file(file_path: str) -> bool:
    """
    Validate if a file is a valid audio file.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        # Try to get file info
        info = sf.info(file_path)
        # Basic validation
        if info.duration <= 0 or info.samplerate <= 0:
            return False
        return True
    except Exception:
        return False


# Supported audio formats
SUPPORTED_FORMATS = {
    '.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', 
    '.wma', '.aiff', '.au', '.mp4'
}


def get_supported_formats() -> set:
    """
    Get the set of supported audio formats.
    
    Returns:
        set: Set of supported file extensions
    """
    return SUPPORTED_FORMATS.copy()


if __name__ == "__main__":
    # Example usage
    print("Audio Utilities Module")
    print(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
