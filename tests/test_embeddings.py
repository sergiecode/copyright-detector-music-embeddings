"""
Test suite for embedding extraction functionality.

Tests all functionality in src/embeddings.py including embedding models,
extraction methods, and comparison functions.

Author: Sergie Code
"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

from . import TestBase, create_test_audio
from embeddings import (
    AudioEmbeddingExtractor, compare_embeddings,
    SimpleSpectrogramExtractor, BaseEmbeddingExtractor
)


class TestBaseEmbeddingExtractor:
    """Test the base embedding extractor interface."""
    
    def test_base_extractor_is_abstract(self):
        """Test that base extractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbeddingExtractor()
        print("✅ Base extractor abstract test passed")


class TestSimpleSpectrogramExtractor(TestBase):
    """Test the simple spectrogram-based embedding extractor."""
    
    def test_initialization(self):
        """Test spectrogram extractor initialization."""
        extractor = SimpleSpectrogramExtractor()
        
        assert extractor.n_mels == 128
        assert extractor.target_sr == 22050
        print("✅ Spectrogram extractor initialization test passed")
    
    def test_extract_embeddings_from_array(self):
        """Test embedding extraction from audio array."""
        extractor = SimpleSpectrogramExtractor()
        
        embedding = extractor.extract_embeddings_from_array(
            self.test_audio, 
            self.test_sr
        )
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # Should be 1D vector
        assert embedding.shape[0] > 0
        assert not np.any(np.isnan(embedding))
        print("✅ Spectrogram embedding extraction test passed")
    
    def test_extract_embeddings_from_file(self):
        """Test embedding extraction from audio file."""
        if self.test_files:
            extractor = SimpleSpectrogramExtractor()
            
            embedding = extractor.extract_embeddings(self.test_files[0])
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] > 0
            print("✅ Spectrogram file extraction test passed")
    
    def test_extract_embeddings_nonexistent_file(self):
        """Test extraction from non-existent file raises error."""
        extractor = SimpleSpectrogramExtractor()
        
        with pytest.raises(FileNotFoundError):
            extractor.extract_embeddings("nonexistent_file.wav")
        print("✅ Non-existent file extraction test passed")
    
    def test_different_sample_rates(self):
        """Test extraction with different sample rates."""
        extractor = SimpleSpectrogramExtractor(target_sr=16000)
        
        # Test with different input sample rate
        test_audio_44k, _ = create_test_audio(duration=2.0, sample_rate=44100)
        embedding = extractor.extract_embeddings_from_array(test_audio_44k, 44100)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0
        print("✅ Different sample rates test passed")


class TestAudioEmbeddingExtractor(TestBase):
    """Test the main audio embedding extractor."""
    
    def test_initialization_spectrogram(self):
        """Test extractor initialization with spectrogram model."""
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        assert extractor.model_name == 'spectrogram'
        assert isinstance(extractor.extractor, SimpleSpectrogramExtractor)
        print("✅ Main extractor initialization test passed")
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = AudioEmbeddingExtractor.get_available_models()
        
        assert isinstance(models, dict)
        assert 'spectrogram' in models
        assert 'openl3' in models
        assert 'audioclip' in models
        print("✅ Available models test passed")
    
    def test_extract_embeddings_single_file(self):
        """Test single file embedding extraction."""
        if self.test_files:
            extractor = AudioEmbeddingExtractor(model_name='spectrogram')
            
            embedding = extractor.extract_embeddings(self.test_files[0])
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] > 0
            print("✅ Single file extraction test passed")
    
    def test_extract_embeddings_batch(self):
        """Test batch embedding extraction."""
        if len(self.test_files) >= 2:
            extractor = AudioEmbeddingExtractor(model_name='spectrogram')
            
            embeddings = extractor.extract_embeddings_batch(self.test_files[:2])
            
            assert isinstance(embeddings, dict)
            assert len(embeddings) == 2
            
            # Check that embeddings were extracted
            successful = sum(1 for v in embeddings.values() if v is not None)
            assert successful > 0
            print("✅ Batch extraction test passed")
    
    def test_get_embedding_info(self):
        """Test getting embedding model information."""
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        info = extractor.get_embedding_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'embedding_dimension' in info
        assert info['model_name'] == 'spectrogram'
        assert info['embedding_dimension'] > 0
        print("✅ Embedding info test passed")
    
    def test_unknown_model(self):
        """Test initialization with unknown model raises error."""
        with pytest.raises(ValueError):
            AudioEmbeddingExtractor(model_name='unknown_model')
        print("✅ Unknown model test passed")

    @patch('embeddings.OPENL3_AVAILABLE', False)
    def test_openl3_fallback(self):
        """Test fallback when OpenL3 is not available."""
        extractor = AudioEmbeddingExtractor(model_name='openl3')
        
        # Should fallback to spectrogram
        assert isinstance(extractor.extractor, SimpleSpectrogramExtractor)
        print("✅ OpenL3 fallback test passed")

    @patch('embeddings.AUDIOCLIP_AVAILABLE', False)
    def test_audioclip_fallback(self):
        """Test fallback when AudioCLIP is not available."""
        extractor = AudioEmbeddingExtractor(model_name='audioclip')
        
        # Should fallback to spectrogram
        assert isinstance(extractor.extractor, SimpleSpectrogramExtractor)
        print("✅ AudioCLIP fallback test passed")


class TestEmbeddingComparison:
    """Test embedding comparison functions."""
    
    def setUp(self):
        """Set up test embeddings."""
        # Create test embeddings
        self.embedding1 = np.random.randn(100)
        self.embedding2 = np.random.randn(100)
        
        # Create identical embeddings for perfect similarity test
        self.identical_embedding = self.embedding1.copy()
        
        # Create orthogonal embeddings for minimum similarity test
        self.orthogonal_embedding = np.random.randn(100)
        self.orthogonal_embedding = self.orthogonal_embedding - np.dot(
            self.orthogonal_embedding, self.embedding1
        ) * self.embedding1 / np.dot(self.embedding1, self.embedding1)
        self.orthogonal_embedding = self.orthogonal_embedding / np.linalg.norm(self.orthogonal_embedding)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        self.setUp()
        
        # Test identical embeddings
        sim = compare_embeddings(self.embedding1, self.identical_embedding, method='cosine')
        assert abs(sim - 1.0) < 1e-10
        
        # Test orthogonal embeddings
        sim = compare_embeddings(self.embedding1, self.orthogonal_embedding, method='cosine')
        assert abs(sim) < 1e-10
        
        # Test different embeddings
        sim = compare_embeddings(self.embedding1, self.embedding2, method='cosine')
        assert -1 <= sim <= 1
        
        print("✅ Cosine similarity test passed")
    
    def test_euclidean_similarity(self):
        """Test euclidean distance-based similarity."""
        self.setUp()
        
        # Test identical embeddings
        sim = compare_embeddings(self.embedding1, self.identical_embedding, method='euclidean')
        assert sim > 0.5  # Should be high similarity
        
        # Test different embeddings
        sim = compare_embeddings(self.embedding1, self.embedding2, method='euclidean')
        assert 0 <= sim <= 1
        
        print("✅ Euclidean similarity test passed")
    
    def test_correlation_similarity(self):
        """Test correlation-based similarity."""
        self.setUp()
        
        # Test identical embeddings
        sim = compare_embeddings(self.embedding1, self.identical_embedding, method='correlation')
        assert abs(sim - 1.0) < 1e-10
        
        # Test different embeddings
        sim = compare_embeddings(self.embedding1, self.embedding2, method='correlation')
        assert -1 <= sim <= 1
        
        print("✅ Correlation similarity test passed")
    
    def test_unknown_similarity_method(self):
        """Test unknown similarity method raises error."""
        self.setUp()
        
        with pytest.raises(ValueError):
            compare_embeddings(self.embedding1, self.embedding2, method='unknown')
        print("✅ Unknown similarity method test passed")


class TestEmbeddingConsistency(TestBase):
    """Test consistency of embedding extraction."""
    
    def test_embedding_consistency(self):
        """Test that same audio produces same embeddings."""
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Extract embedding twice from same audio
        embedding1 = extractor.extract_embeddings_from_array(self.test_audio, self.test_sr)
        embedding2 = extractor.extract_embeddings_from_array(self.test_audio, self.test_sr)
        
        # Should be identical
        assert np.allclose(embedding1, embedding2)
        print("✅ Embedding consistency test passed")
    
    def test_different_audio_different_embeddings(self):
        """Test that different audio produces different embeddings."""
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Create two different audio signals
        audio1, sr = create_test_audio(frequency=440.0)
        audio2, sr = create_test_audio(frequency=880.0)
        
        embedding1 = extractor.extract_embeddings_from_array(audio1, sr)
        embedding2 = extractor.extract_embeddings_from_array(audio2, sr)
        
        # Should be different
        assert not np.allclose(embedding1, embedding2)
        print("✅ Different audio test passed")


def test_embeddings_module_import():
    """Test that embeddings module imports correctly."""
    try:
        import embeddings
        print("✅ Embeddings module import test passed")
    except ImportError as e:
        pytest.fail(f"Failed to import embeddings module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
