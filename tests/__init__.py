"""
Test suite initialization for music embeddings project.

This module sets up the testing environment and provides common utilities
for testing audio embedding extraction functionality.

Author: Sergie Code
"""

import os
import sys
import numpy as np
import tempfile
import shutil
from typing import Tuple

# Add src directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def create_test_audio(duration: float = 2.0, 
                     sample_rate: int = 22050,
                     frequency: float = 440.0) -> Tuple[np.ndarray, int]:
    """
    Create synthetic test audio data.
    
    Args:
        duration (float): Duration in seconds
        sample_rate (int): Sample rate in Hz
        frequency (float): Frequency in Hz
    
    Returns:
        Tuple[np.ndarray, int]: Audio array and sample rate
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio, sample_rate

def create_test_audio_files(temp_dir: str) -> list:
    """
    Create temporary test audio files.
    
    Args:
        temp_dir (str): Temporary directory path
    
    Returns:
        list: List of created audio file paths
    """
    import soundfile as sf
    
    test_files = []
    
    # Create different types of test audio
    test_configs = [
        ('test_sine_440.wav', 2.0, 440.0),
        ('test_sine_880.wav', 2.0, 880.0),
        ('test_chirp.wav', 3.0, None),  # Special case for chirp
        ('test_noise.wav', 1.5, None)   # Special case for noise
    ]
    
    for filename, duration, freq in test_configs:
        filepath = os.path.join(temp_dir, filename)
        
        if 'chirp' in filename:
            # Create frequency sweep
            t = np.linspace(0, duration, int(22050 * duration))
            freq_sweep = 200 + (2000 - 200) * t / duration
            audio = 0.3 * np.sin(2 * np.pi * freq_sweep * t)
        elif 'noise' in filename:
            # Create pink noise
            audio = np.random.normal(0, 0.1, int(22050 * duration))
        else:
            # Create sine wave
            audio, _ = create_test_audio(duration, 22050, freq)
        
        sf.write(filepath, audio, 22050)
        test_files.append(filepath)
    
    return test_files

class TestBase:
    """Base class for test cases with common setup and teardown."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with temporary directory and test files."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_files = create_test_audio_files(cls.temp_dir)
        cls.test_audio, cls.test_sr = create_test_audio()
    
    @classmethod
    def teardown_class(cls):
        """Clean up temporary files and directories."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

# Test data constants
TEST_SAMPLE_RATES = [16000, 22050, 44100, 48000]
TEST_DURATIONS = [1.0, 2.5, 5.0]
TEST_FREQUENCIES = [220.0, 440.0, 880.0, 1760.0]
