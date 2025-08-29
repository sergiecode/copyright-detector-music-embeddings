"""
Audio Embeddings Extraction Module

This module provides the main functionality for extracting embeddings
from audio files using various pre-trained models.

Author: Sergie Code
"""

import os
import numpy as np
import librosa
import torch
from typing import Dict, Any, List
import warnings
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import model-specific libraries
try:
    import openl3
    OPENL3_AVAILABLE = True
except ImportError:
    OPENL3_AVAILABLE = False
    print("⚠️ OpenL3 not available. Install with: pip install openl3")

try:
    from transformers import AutoModel, AutoProcessor
    AUDIOCLIP_AVAILABLE = True
except ImportError:
    AUDIOCLIP_AVAILABLE = False
    print("⚠️ AudioCLIP dependencies not available. Install with: pip install transformers")


class BaseEmbeddingExtractor(ABC):
    """Abstract base class for audio embedding extractors."""
    
    @abstractmethod
    def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """Extract embeddings from an audio file."""
        pass
    
    @abstractmethod
    def extract_embeddings_from_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract embeddings from an audio array."""
        pass


class OpenL3Extractor(BaseEmbeddingExtractor):
    """OpenL3 embedding extractor."""
    
    def __init__(self, 
                 input_repr: str = 'mel256',
                 content_type: str = 'music',
                 embedding_size: int = 6144):
        """
        Initialize OpenL3 extractor.
        
        Args:
            input_repr (str): Input representation ('linear', 'mel128', 'mel256')
            content_type (str): Content type ('music' or 'env')
            embedding_size (int): Size of embeddings (512 or 6144)
        """
        if not OPENL3_AVAILABLE:
            raise ImportError("OpenL3 is not available. Install with: pip install openl3")
        
        self.input_repr = input_repr
        self.content_type = content_type
        self.embedding_size = embedding_size
        
        print(f"✓ Initialized OpenL3 extractor:")
        print(f"  - Input representation: {input_repr}")
        print(f"  - Content type: {content_type}")
        print(f"  - Embedding size: {embedding_size}")
    
    def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """Extract OpenL3 embeddings from an audio file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio with the right sample rate for OpenL3
            audio, sr = librosa.load(audio_path, sr=48000)  # OpenL3 expects 48kHz
            
            return self.extract_embeddings_from_array(audio, sr)
            
        except Exception as e:
            raise Exception(f"Error extracting OpenL3 embeddings: {str(e)}")
    
    def extract_embeddings_from_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract OpenL3 embeddings from an audio array."""
        try:
            # Ensure correct sample rate
            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000
            
            # Extract embeddings
            embeddings, timestamps = openl3.get_audio_embedding(
                audio,
                sr,
                input_repr=self.input_repr,
                content_type=self.content_type,
                embedding_size=self.embedding_size,
                verbose=False
            )
            
            # Average across time to get a single embedding vector
            embedding_vector = np.mean(embeddings, axis=0)
            
            print(f"✓ Extracted OpenL3 embeddings")
            print(f"  - Shape: {embedding_vector.shape}")
            print(f"  - Temporal frames: {embeddings.shape[0]}")
            
            return embedding_vector
            
        except Exception as e:
            raise Exception(f"Error processing audio with OpenL3: {str(e)}")


class AudioCLIPExtractor(BaseEmbeddingExtractor):
    """AudioCLIP embedding extractor."""
    
    def __init__(self, model_name: str = "microsoft/unispeech-large"):
        """
        Initialize AudioCLIP extractor.
        
        Args:
            model_name (str): Name of the pre-trained model
        """
        if not AUDIOCLIP_AVAILABLE:
            raise ImportError("AudioCLIP dependencies not available. Install transformers package.")
        
        self.model_name = model_name
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            print(f"✓ Initialized AudioCLIP extractor:")
            print(f"  - Model: {model_name}")
            
        except Exception as e:
            raise Exception(f"Error loading AudioCLIP model: {str(e)}")
    
    def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """Extract AudioCLIP embeddings from an audio file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)  # Most transformers expect 16kHz
            
            return self.extract_embeddings_from_array(audio, sr)
            
        except Exception as e:
            raise Exception(f"Error extracting AudioCLIP embeddings: {str(e)}")
    
    def extract_embeddings_from_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract AudioCLIP embeddings from an audio array."""
        try:
            # Ensure correct sample rate
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Process audio
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state
                
                # Pool to get a single vector (mean pooling)
                embedding_vector = torch.mean(embeddings, dim=1).squeeze().numpy()
            
            print(f"✓ Extracted AudioCLIP embeddings")
            print(f"  - Shape: {embedding_vector.shape}")
            
            return embedding_vector
            
        except Exception as e:
            raise Exception(f"Error processing audio with AudioCLIP: {str(e)}")


class SimpleSpectrogramExtractor(BaseEmbeddingExtractor):
    """Simple spectrogram-based embedding extractor for fallback."""
    
    def __init__(self, 
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 target_sr: int = 22050):
        """
        Initialize simple spectrogram extractor.
        
        Args:
            n_mels (int): Number of mel frequency bins
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            target_sr (int): Target sample rate
        """
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_sr = target_sr
        
        print(f"✓ Initialized Simple Spectrogram extractor:")
        print(f"  - Mel bins: {n_mels}")
        print(f"  - Sample rate: {target_sr}")
    
    def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """Extract spectrogram-based embeddings from an audio file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            return self.extract_embeddings_from_array(audio, sr)
            
        except Exception as e:
            raise Exception(f"Error extracting spectrogram embeddings: {str(e)}")
    
    def extract_embeddings_from_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectrogram-based embeddings from an audio array."""
        try:
            # Try librosa first if available
            try:
                import librosa
                
                # Resample if needed
                if sr != self.target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                
                # Extract mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.target_sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                
                # Convert to log scale
                log_mel_spec = librosa.power_to_db(mel_spec)
                
                # Create embedding by taking statistics across time
                embedding_vector = np.concatenate([
                    np.mean(log_mel_spec, axis=1),  # Mean across time
                    np.std(log_mel_spec, axis=1),   # Std across time
                    np.min(log_mel_spec, axis=1),   # Min across time
                    np.max(log_mel_spec, axis=1)    # Max across time
                ])
                
                print(f"✓ Extracted Spectrogram embeddings")
                print(f"  - Shape: {embedding_vector.shape}")
                
                return embedding_vector
                
            except (ImportError, Exception) as librosa_error:
                print(f"⚠️ Librosa fallback due to: {str(librosa_error)}")
                return self._extract_basic_features(audio, sr)
            
        except Exception as e:
            raise Exception(f"Error processing audio for spectrogram: {str(e)}")
    
    def _extract_basic_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract basic audio features without librosa dependencies."""
        try:
            # Basic time domain features
            features = []
            
            # 1. Statistical features
            features.extend([
                np.mean(audio),              # Mean amplitude
                np.std(audio),               # Standard deviation  
                np.max(np.abs(audio)),       # Peak amplitude
                np.mean(np.abs(audio)),      # Mean absolute amplitude
                np.var(audio),               # Variance
                np.median(audio),            # Median
                np.percentile(audio, 25),    # 25th percentile
                np.percentile(audio, 75),    # 75th percentile
            ])
            
            # 2. Zero crossing rate
            zero_crossings = len(np.where(np.diff(np.sign(audio)))[0])
            zcr = zero_crossings / len(audio)
            features.append(zcr)
            
            # 3. Simple spectral features using FFT
            try:
                # Compute FFT
                fft = np.fft.fft(audio)
                magnitude = np.abs(fft[:len(fft)//2])
                
                # Spectral features
                features.extend([
                    np.mean(magnitude),      # Spectral mean
                    np.std(magnitude),       # Spectral std
                    np.max(magnitude),       # Spectral peak
                    np.sum(magnitude),       # Spectral energy
                ])
                
                # Frequency bins (simplified mel-like)
                n_bins = self.n_mels // 4  # Use fewer bins
                bin_size = len(magnitude) // n_bins
                
                for i in range(n_bins):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, len(magnitude))
                    if end_idx > start_idx:
                        bin_energy = np.mean(magnitude[start_idx:end_idx])
                        features.append(bin_energy)
                    else:
                        features.append(0.0)
                        
            except Exception:
                # If FFT fails, pad with zeros
                features.extend([0.0] * (self.n_mels // 4 + 4))
            
            # Ensure we have the right number of features
            target_length = self.n_mels
            current_length = len(features)
            
            if current_length < target_length:
                # Pad with repeated patterns or zeros
                padding_needed = target_length - current_length
                if current_length > 0:
                    # Repeat pattern
                    repeat_pattern = features * (padding_needed // current_length + 1)
                    features.extend(repeat_pattern[:padding_needed])
                else:
                    features.extend([0.0] * padding_needed)
            elif current_length > target_length:
                features = features[:target_length]
            
            embedding_vector = np.array(features, dtype=np.float32)
            
            print(f"✓ Extracted Basic Audio Features (fallback)")
            print(f"  - Shape: {embedding_vector.shape}")
            
            return embedding_vector
            
        except Exception as e:
            # Ultimate fallback - return zeros with correct shape
            print(f"⚠️ Using zero fallback due to: {str(e)}")
            return np.zeros(self.n_mels, dtype=np.float32)


class AudioEmbeddingExtractor:
    """Main class for extracting audio embeddings using various models."""
    
    AVAILABLE_MODELS = {
        'openl3': 'OpenL3 - Look, Listen and Learn',
        'audioclip': 'AudioCLIP - Audio-text embeddings',
        'spectrogram': 'Simple Spectrogram - Fallback method'
    }
    
    def __init__(self, 
                 model_name: str = 'openl3',
                 **model_kwargs):
        """
        Initialize the audio embedding extractor.
        
        Args:
            model_name (str): Name of the model to use
            **model_kwargs: Additional arguments for the specific model
        """
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        
        # Initialize the appropriate extractor
        self.extractor = self._initialize_extractor()
        
        print(f"🎵 AudioEmbeddingExtractor ready!")
        print(f"   Model: {self.AVAILABLE_MODELS.get(model_name, model_name)}")
    
    def _initialize_extractor(self) -> BaseEmbeddingExtractor:
        """Initialize the appropriate embedding extractor."""
        if self.model_name == 'openl3':
            if not OPENL3_AVAILABLE:
                print("⚠️ OpenL3 not available, falling back to spectrogram")
                return SimpleSpectrogramExtractor(**self.model_kwargs)
            return OpenL3Extractor(**self.model_kwargs)
            
        elif self.model_name == 'audioclip':
            if not AUDIOCLIP_AVAILABLE:
                print("⚠️ AudioCLIP not available, falling back to spectrogram")
                return SimpleSpectrogramExtractor(**self.model_kwargs)
            return AudioCLIPExtractor(**self.model_kwargs)
            
        elif self.model_name == 'spectrogram':
            return SimpleSpectrogramExtractor(**self.model_kwargs)
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}. "
                           f"Available models: {list(self.AVAILABLE_MODELS.keys())}")
    
    def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """
        Extract embeddings from an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Embedding vector
        """
        print(f"🎵 Extracting embeddings from: {os.path.basename(audio_path)}")
        return self.extractor.extract_embeddings(audio_path)
    
    def extract_embeddings_from_array(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract embeddings from an audio array.
        
        Args:
            audio (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            np.ndarray: Embedding vector
        """
        return self.extractor.extract_embeddings_from_array(audio, sr)
    
    def extract_embeddings_batch(self, audio_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from multiple audio files.
        
        Args:
            audio_paths (List[str]): List of audio file paths
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping file paths to embeddings
        """
        embeddings = {}
        
        for i, audio_path in enumerate(audio_paths):
            try:
                print(f"\n📁 Processing file {i+1}/{len(audio_paths)}")
                embedding = self.extract_embeddings(audio_path)
                embeddings[audio_path] = embedding
                
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {str(e)}")
                embeddings[audio_path] = None
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Successfully processed: {sum(1 for v in embeddings.values() if v is not None)}/{len(audio_paths)}")
        
        return embeddings
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get information about available models."""
        available = {}
        
        for model, description in cls.AVAILABLE_MODELS.items():
            if model == 'openl3' and OPENL3_AVAILABLE:
                available[model] = description + " ✓"
            elif model == 'audioclip' and AUDIOCLIP_AVAILABLE:
                available[model] = description + " ✓"
            elif model == 'spectrogram':
                available[model] = description + " ✓"
            else:
                available[model] = description + " ❌"
        
        return available
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the current model and embedding dimensions."""
        # Extract a dummy embedding to get dimensions
        dummy_audio = np.random.randn(22050)  # 1 second of random audio
        dummy_embedding = self.extractor.extract_embeddings_from_array(dummy_audio, 22050)
        
        return {
            'model_name': self.model_name,
            'model_description': self.AVAILABLE_MODELS.get(self.model_name),
            'embedding_dimension': dummy_embedding.shape[0],
            'embedding_dtype': str(dummy_embedding.dtype)
        }


def compare_embeddings(embedding1: np.ndarray, 
                      embedding2: np.ndarray,
                      method: str = 'cosine') -> float:
    """
    Compare two embeddings using various similarity metrics.
    
    Args:
        embedding1 (np.ndarray): First embedding
        embedding2 (np.ndarray): Second embedding
        method (str): Similarity method ('cosine', 'euclidean', 'correlation')
        
    Returns:
        float: Similarity score
    """
    if method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
        
    elif method == 'euclidean':
        # Euclidean distance (converted to similarity)
        distance = np.linalg.norm(embedding1 - embedding2)
        return 1 / (1 + distance)  # Convert distance to similarity
        
    elif method == 'correlation':
        # Pearson correlation
        return np.corrcoef(embedding1, embedding2)[0, 1]
        
    else:
        raise ValueError(f"Unknown similarity method: {method}")


if __name__ == "__main__":
    # Example usage
    print("🎵 Music Embeddings Extraction Module")
    print("\nAvailable models:")
    for model, description in AudioEmbeddingExtractor.get_available_models().items():
        print(f"  - {model}: {description}")
    
    # Quick test with dummy data
    print("\n🧪 Testing with dummy audio...")
    extractor = AudioEmbeddingExtractor(model_name='spectrogram')
    dummy_audio = np.random.randn(44100)  # 2 seconds at 22050 Hz
    embedding = extractor.extract_embeddings_from_array(dummy_audio, 22050)
    print(f"✓ Test successful! Embedding shape: {embedding.shape}")
