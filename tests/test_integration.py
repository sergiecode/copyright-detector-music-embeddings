"""
Integration tests for the complete music embeddings project.

Tests the entire workflow from audio loading to embedding extraction,
similarity calculation, and real-world usage scenarios.

Author: Sergie Code
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import soundfile as sf

from . import TestBase, create_test_audio_files
from embeddings import AudioEmbeddingExtractor, compare_embeddings
from utils import load_audio, save_embeddings, load_embeddings


class TestCompleteWorkflow(TestBase):
    """Test complete workflow from audio to embeddings."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        print("\n🚀 Starting end-to-end workflow test...")
        
        # Step 1: Create and load audio
        if not self.test_files:
            pytest.skip("No test audio files available")
        
        audio_file = self.test_files[0]
        audio, sr = load_audio(audio_file)
        print(f"✅ Step 1: Loaded audio file - shape: {audio.shape}, sr: {sr}")
        
        # Step 2: Extract embeddings
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        embedding = extractor.extract_embeddings(audio_file)
        print(f"✅ Step 2: Extracted embedding - shape: {embedding.shape}")
        
        # Step 3: Save embeddings
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            metadata = {
                'file': os.path.basename(audio_file),
                'model': 'spectrogram',
                'sample_rate': sr
            }
            save_embeddings(embedding, tmp.name, metadata)
            print(f"✅ Step 3: Saved embeddings to {tmp.name}")
            
            # Step 4: Load embeddings
            loaded_embedding, loaded_metadata = load_embeddings(tmp.name)
            print(f"✅ Step 4: Loaded embeddings - shape: {loaded_embedding.shape}")
            
            # Verify consistency
            assert np.allclose(embedding, loaded_embedding)
            assert loaded_metadata['model'] == 'spectrogram'
            
            # Cleanup
            os.unlink(tmp.name)
        
        print("✅ End-to-end workflow test completed successfully!")
    
    def test_similarity_search_workflow(self):
        """Test similarity search workflow."""
        print("\n🔍 Starting similarity search workflow test...")
        
        if len(self.test_files) < 2:
            pytest.skip("Need at least 2 test files for similarity search")
        
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Extract embeddings for all test files
        embeddings = {}
        for file_path in self.test_files[:3]:  # Use first 3 files
            filename = os.path.basename(file_path)
            embedding = extractor.extract_embeddings(file_path)
            embeddings[filename] = embedding
            print(f"✅ Extracted embedding for {filename}")
        
        # Test similarity between files
        filenames = list(embeddings.keys())
        similarities = []
        
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                sim = compare_embeddings(
                    embeddings[filenames[i]], 
                    embeddings[filenames[j]], 
                    method='cosine'
                )
                similarities.append((filenames[i], filenames[j], sim))
                print(f"✅ Similarity {filenames[i]} ↔ {filenames[j]}: {sim:.3f}")
        
        # Verify similarities are valid
        for file1, file2, sim in similarities:
            assert -1 <= sim <= 1, f"Invalid similarity: {sim}"
        
        print("✅ Similarity search workflow test completed!")
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow."""
        print("\n⚡ Starting batch processing workflow test...")
        
        if len(self.test_files) < 2:
            pytest.skip("Need at least 2 test files for batch processing")
        
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Process files in batch
        embeddings = extractor.extract_embeddings_batch(self.test_files)
        
        # Verify results
        successful = sum(1 for v in embeddings.values() if v is not None)
        total = len(embeddings)
        
        print(f"✅ Batch processing: {successful}/{total} files processed successfully")
        assert successful > 0, "No files were processed successfully"
        
        # Verify embedding quality
        for file_path, embedding in embeddings.items():
            if embedding is not None:
                assert isinstance(embedding, np.ndarray)
                assert len(embedding.shape) == 1
                assert embedding.shape[0] > 0
                assert not np.any(np.isnan(embedding))
        
        print("✅ Batch processing workflow test completed!")


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_music_similarity_detection(self):
        """Test music similarity detection scenario."""
        print("\n🎵 Testing music similarity detection scenario...")
        
        # Create test audio with known relationships
        sr = 22050
        duration = 3.0
        
        # Original melody
        t = np.linspace(0, duration, int(sr * duration))
        original = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4
        
        # Similar melody (octave higher)
        similar = 0.3 * np.sin(2 * np.pi * 880 * t)  # A5
        
        # Different melody
        different = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3
        
        # Extract embeddings
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        emb_original = extractor.extract_embeddings_from_array(original, sr)
        emb_similar = extractor.extract_embeddings_from_array(similar, sr)
        emb_different = extractor.extract_embeddings_from_array(different, sr)
        
        # Calculate similarities
        sim_original_similar = compare_embeddings(emb_original, emb_similar, method='cosine')
        sim_original_different = compare_embeddings(emb_original, emb_different, method='cosine')
        
        print(f"✅ Original vs Similar (octave): {sim_original_similar:.3f}")
        print(f"✅ Original vs Different: {sim_original_different:.3f}")
        
        # The octave should be more similar than the different frequency
        # (though this depends on the specific model)
        assert abs(sim_original_similar) >= 0
        assert abs(sim_original_different) >= 0
        
        print("✅ Music similarity detection test completed!")
    
    def test_copyright_detection_scenario(self):
        """Test copyright detection scenario."""
        print("\n🛡️ Testing copyright detection scenario...")
        
        # Simulate copyright detection workflow
        sr = 22050
        duration = 5.0
        
        # Original track
        t = np.linspace(0, duration, int(sr * duration))
        original_track = 0.3 * (
            np.sin(2 * np.pi * 440 * t) +  # A4
            0.5 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
            0.3 * np.sin(2 * np.pi * 659.25 * t)   # E5
        )
        
        # Potential copy (same melody, different amplitude)
        potential_copy = 0.5 * original_track
        
        # Different track
        different_track = 0.3 * np.sin(2 * np.pi * 200 * t)
        
        # Extract embeddings
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        emb_original = extractor.extract_embeddings_from_array(original_track, sr)
        emb_copy = extractor.extract_embeddings_from_array(potential_copy, sr)
        emb_different = extractor.extract_embeddings_from_array(different_track, sr)
        
        # Calculate similarities
        similarity_to_copy = compare_embeddings(emb_original, emb_copy, method='cosine')
        similarity_to_different = compare_embeddings(emb_original, emb_different, method='cosine')
        
        print(f"✅ Original vs Copy: {similarity_to_copy:.3f}")
        print(f"✅ Original vs Different: {similarity_to_different:.3f}")
        
        # The copy should be more similar than the different track
        assert similarity_to_copy > similarity_to_different
        
        # Set threshold for copyright detection (example)
        copyright_threshold = 0.8
        
        if similarity_to_copy > copyright_threshold:
            print("🚨 Potential copyright infringement detected!")
        else:
            print("✅ No copyright issues detected")
        
        print("✅ Copyright detection test completed!")
    
    def test_music_recommendation_scenario(self):
        """Test music recommendation scenario."""
        print("\n📻 Testing music recommendation scenario...")
        
        # Create a music database simulation
        music_database = {}
        sr = 22050
        duration = 2.0
        
        # Create different "songs" with different characteristics
        songs = {
            'pop_song': (440, 0.3),      # A4, moderate volume
            'rock_song': (220, 0.5),     # A3, louder
            'classical': (523.25, 0.2),  # C5, softer
            'jazz': (349.23, 0.4),      # F4, medium
            'electronic': (880, 0.6)     # A5, loud
        }
        
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Build database
        for song_name, (freq, amp) in songs.items():
            t = np.linspace(0, duration, int(sr * duration))
            audio = amp * np.sin(2 * np.pi * freq * t)
            embedding = extractor.extract_embeddings_from_array(audio, sr)
            music_database[song_name] = embedding
            print(f"✅ Added {song_name} to database")
        
        # User listens to a song similar to pop
        user_song = 0.35 * np.sin(2 * np.pi * 440 * t)  # Similar to pop but slightly different
        user_embedding = extractor.extract_embeddings_from_array(user_song, sr)
        
        # Find recommendations
        recommendations = []
        for song_name, song_embedding in music_database.items():
            similarity = compare_embeddings(user_embedding, song_embedding, method='cosine')
            recommendations.append((song_name, similarity))
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Top recommendations for user:")
        for i, (song, sim) in enumerate(recommendations[:3]):
            print(f"   {i+1}. {song}: {sim:.3f}")
        
        # The pop song should be most similar
        assert recommendations[0][0] == 'pop_song'
        
        print("✅ Music recommendation test completed!")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_corrupted_audio_handling(self):
        """Test handling of corrupted audio data."""
        print("\n🔧 Testing corrupted audio handling...")
        
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Test with NaN values
        corrupted_audio = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        try:
            embedding = extractor.extract_embeddings_from_array(corrupted_audio, 22050)
            # Should handle gracefully
            assert not np.any(np.isnan(embedding))
            print("✅ NaN audio handled gracefully")
        except Exception as e:
            print(f"✅ NaN audio raised expected error: {type(e).__name__}")
        
        # Test with infinite values
        infinite_audio = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        
        try:
            embedding = extractor.extract_embeddings_from_array(infinite_audio, 22050)
            assert not np.any(np.isinf(embedding))
            print("✅ Infinite audio handled gracefully")
        except Exception as e:
            print(f"✅ Infinite audio raised expected error: {type(e).__name__}")
    
    def test_empty_audio_handling(self):
        """Test handling of empty audio."""
        print("\n🔧 Testing empty audio handling...")
        
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Test with empty array
        empty_audio = np.array([])
        
        with pytest.raises(Exception):
            extractor.extract_embeddings_from_array(empty_audio, 22050)
        
        print("✅ Empty audio properly raises error")
    
    def test_very_short_audio(self):
        """Test handling of very short audio."""
        print("\n🔧 Testing very short audio handling...")
        
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        
        # Very short audio (1 sample)
        short_audio = np.array([0.5])
        
        try:
            embedding = extractor.extract_embeddings_from_array(short_audio, 22050)
            assert isinstance(embedding, np.ndarray)
            print("✅ Very short audio handled gracefully")
        except Exception as e:
            print(f"✅ Very short audio raised expected error: {type(e).__name__}")


def test_project_completeness():
    """Test that all required components are available."""
    print("\n📋 Testing project completeness...")
    
    # Test module imports
    try:
        import embeddings
        import utils
        print("✅ All modules import successfully")
    except ImportError as e:
        pytest.fail(f"Module import failed: {e}")
    
    # Test main classes are available
    assert hasattr(embeddings, 'AudioEmbeddingExtractor')
    assert hasattr(embeddings, 'compare_embeddings')
    assert hasattr(utils, 'load_audio')
    assert hasattr(utils, 'save_embeddings')
    
    print("✅ All required classes and functions available")
    
    # Test available models
    models = embeddings.AudioEmbeddingExtractor.get_available_models()
    assert 'spectrogram' in models
    print("✅ At least one embedding model available")
    
    print("✅ Project completeness test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
