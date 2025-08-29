"""
Test suite for audio utilities module.

Tests all functionality in src/utils.py including audio loading,
preprocessing, embedding management, and helper functions.

Author: Sergie Code
"""

import pytest
import numpy as np
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

from . import TestBase, create_test_audio, create_test_audio_files
from utils import (
    load_audio, preprocess_audio, save_embeddings, load_embeddings,
    get_audio_info, create_audio_segments, validate_audio_file,
    get_supported_formats
)


class TestAudioLoading(TestBase):
    """Test audio loading functionality."""
    
    def test_load_audio_basic(self):
        """Test basic audio loading functionality."""
        if self.test_files:
            audio, sr = load_audio(self.test_files[0])
            
            assert isinstance(audio, np.ndarray)
            assert isinstance(sr, int)
            assert len(audio) > 0
            assert sr > 0
            print(f"✅ Basic audio loading test passed")
    
    def test_load_audio_with_target_sr(self):
        """Test audio loading with target sample rate."""
        if self.test_files:
            target_sr = 16000
            audio, sr = load_audio(self.test_files[0], target_sr=target_sr)
            
            assert sr == target_sr
            assert len(audio) > 0
            print(f"✅ Target sample rate test passed")
    
    def test_load_audio_with_duration(self):
        """Test audio loading with duration limit."""
        if self.test_files:
            duration = 1.0
            audio, sr = load_audio(self.test_files[0], duration=duration)
            
            expected_samples = int(sr * duration)
            assert len(audio) <= expected_samples + 1000  # Allow small tolerance
            print(f"✅ Duration limit test passed")
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent_file.wav")
        print(f"✅ Non-existent file test passed")


class TestAudioPreprocessing(TestBase):
    """Test audio preprocessing functionality."""
    
    def test_preprocess_audio_basic(self):
        """Test basic audio preprocessing."""
        processed = preprocess_audio(self.test_audio)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) <= len(self.test_audio)
        assert np.max(np.abs(processed)) <= 1.0  # Should be normalized
        print(f"✅ Basic preprocessing test passed")
    
    def test_preprocess_audio_max_duration(self):
        """Test preprocessing with maximum duration."""
        max_duration = 1.0
        processed = preprocess_audio(
            self.test_audio, 
            target_sr=self.test_sr,
            max_duration=max_duration
        )
        
        expected_samples = int(self.test_sr * max_duration)
        assert len(processed) <= expected_samples
        print(f"✅ Max duration preprocessing test passed")
    
    def test_preprocess_audio_no_normalization(self):
        """Test preprocessing without normalization."""
        # Create audio with known amplitude
        test_audio = np.array([0.5, -0.5, 0.8, -0.8])
        processed = preprocess_audio(test_audio, normalize=False)
        
        # Should not change the amplitude significantly
        assert np.allclose(test_audio, processed, atol=0.1)
        print(f"✅ No normalization test passed")


class TestEmbeddingManagement(TestBase):
    """Test embedding save/load functionality."""
    
    def setUp(self):
        """Set up test embeddings."""
        self.test_embedding = np.random.randn(128)
        self.test_metadata = {
            'model': 'test_model',
            'date': '2025-08-29',
            'config': {'param1': 'value1'}
        }
    
    def test_save_load_pickle(self):
        """Test saving and loading embeddings in pickle format."""
        self.setUp()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            try:
                # Save embedding
                save_embeddings(
                    self.test_embedding, 
                    tmp.name, 
                    self.test_metadata, 
                    format='pickle'
                )
                
                # Load embedding
                loaded_embedding, loaded_metadata = load_embeddings(tmp.name)
                
                # Verify
                assert np.allclose(self.test_embedding, loaded_embedding)
                assert loaded_metadata['model'] == self.test_metadata['model']
                print(f"✅ Pickle save/load test passed")
                
            finally:
                os.unlink(tmp.name)
    
    def test_save_load_npy(self):
        """Test saving and loading embeddings in numpy format."""
        self.setUp()
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            try:
                # Save embedding
                save_embeddings(
                    self.test_embedding, 
                    tmp.name, 
                    self.test_metadata, 
                    format='npy'
                )
                
                # Load embedding
                loaded_embedding, loaded_metadata = load_embeddings(tmp.name)
                
                # Verify
                assert np.allclose(self.test_embedding, loaded_embedding)
                print(f"✅ NumPy save/load test passed")
                
            finally:
                os.unlink(tmp.name)
                # Clean up metadata file
                metadata_file = tmp.name.replace('.npy', '_metadata.json')
                if os.path.exists(metadata_file):
                    os.unlink(metadata_file)
    
    def test_load_nonexistent_embedding(self):
        """Test loading non-existent embedding file."""
        with pytest.raises(FileNotFoundError):
            load_embeddings("nonexistent_embedding.pkl")
        print(f"✅ Non-existent embedding test passed")


class TestAudioInfo(TestBase):
    """Test audio information extraction."""
    
    def test_get_audio_info(self):
        """Test getting audio file information."""
        if self.test_files:
            info = get_audio_info(self.test_files[0])
            
            assert isinstance(info, dict)
            assert 'duration_seconds' in info
            assert 'sample_rate' in info
            assert 'channels' in info
            assert info['duration_seconds'] > 0
            assert info['sample_rate'] > 0
            print(f"✅ Audio info test passed")
    
    def test_get_audio_info_nonexistent(self):
        """Test getting info for non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_audio_info("nonexistent_file.wav")
        print(f"✅ Non-existent audio info test passed")


class TestAudioSegments(TestBase):
    """Test audio segmentation functionality."""
    
    def test_create_audio_segments(self):
        """Test creating audio segments."""
        segment_duration = 1.0
        segments = create_audio_segments(
            self.test_audio, 
            self.test_sr, 
            segment_duration=segment_duration
        )
        
        assert isinstance(segments, list)
        assert len(segments) > 0
        
        expected_samples = int(self.test_sr * segment_duration)
        for segment in segments:
            assert len(segment) == expected_samples
        
        print(f"✅ Audio segments test passed")
    
    def test_create_audio_segments_with_overlap(self):
        """Test creating overlapping audio segments."""
        segments = create_audio_segments(
            self.test_audio,
            self.test_sr,
            segment_duration=1.0,
            overlap=0.5
        )
        
        # With 50% overlap, we should get more segments
        segments_no_overlap = create_audio_segments(
            self.test_audio,
            self.test_sr,
            segment_duration=1.0,
            overlap=0.0
        )
        
        assert len(segments) >= len(segments_no_overlap)
        print(f"✅ Overlapping segments test passed")


class TestAudioValidation(TestBase):
    """Test audio file validation."""
    
    def test_validate_existing_audio_file(self):
        """Test validation of existing audio file."""
        if self.test_files:
            is_valid = validate_audio_file(self.test_files[0])
            assert is_valid
            print(f"✅ Valid audio file test passed")
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        is_valid = validate_audio_file("nonexistent_file.wav")
        assert not is_valid
        print(f"✅ Invalid audio file test passed")
    
    def test_validate_non_audio_file(self):
        """Test validation of non-audio file."""
        # Create a text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"This is not an audio file")
            
        try:
            is_valid = validate_audio_file(tmp.name)
            assert not is_valid
            print(f"✅ Non-audio file test passed")
        finally:
            os.unlink(tmp.name)


class TestSupportedFormats:
    """Test supported formats functionality."""
    
    def test_get_supported_formats(self):
        """Test getting supported audio formats."""
        formats = get_supported_formats()
        
        assert isinstance(formats, set)
        assert len(formats) > 0
        assert '.wav' in formats
        assert '.mp3' in formats
        print(f"✅ Supported formats test passed")


def test_utils_module_import():
    """Test that utils module imports correctly."""
    try:
        import utils
        print(f"✅ Utils module import test passed")
    except ImportError as e:
        pytest.fail(f"Failed to import utils module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
