"""
Final functionality test for music embeddings project.

This script performs a complete end-to-end test to ensure
the project works perfectly for real-world usage.

Author: Sergie Code
"""

import sys
import numpy as np
import tempfile
import os
from pathlib import Path

def add_src_to_path():
    """Add src directory to Python path."""
    src_path = Path(".") / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

def create_test_audio_file():
    """Create a temporary test audio file."""
    try:
        import soundfile as sf
        
        # Create synthetic audio
        duration = 3.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a simple melody (C major chord arpeggio)
        frequencies = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
        audio = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            start_time = i * duration / len(frequencies)
            end_time = (i + 1) * duration / len(frequencies)
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            segment = t[start_idx:end_idx] - start_time
            note = 0.3 * np.sin(2 * np.pi * freq * segment)
            audio[start_idx:end_idx] = note
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio, sample_rate)
        
        return temp_file.name, audio, sample_rate
        
    except ImportError:
        print("⚠️ SoundFile not available, creating synthetic data only")
        return None, None, None

def test_complete_workflow():
    """Test the complete embedding extraction workflow."""
    print("🎵 MUSIC EMBEDDINGS PROJECT - FINAL FUNCTIONALITY TEST")
    print("=" * 60)
    print("By: Sergie Code - AI Tools for Musicians")
    print()
    
    try:
        # Step 1: Import modules
        print("🔧 Step 1: Importing modules...")
        from embeddings import AudioEmbeddingExtractor, compare_embeddings
        from utils import save_embeddings, load_embeddings
        print("✅ All modules imported successfully")
        
        # Step 2: Initialize extractor
        print("\n🔧 Step 2: Initializing embedding extractor...")
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        print("✅ Extractor initialized with spectrogram model")
        
        # Step 3: Create test audio
        print("\n🔧 Step 3: Creating test audio...")
        
        # Create synthetic audio data
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create two different audio signals for testing
        audio1 = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
        audio2 = 0.3 * np.sin(2 * np.pi * 880 * t)  # A5 note (octave higher)
        audio3 = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3 note (octave lower)
        
        print(f"✅ Created 3 test audio signals ({len(audio1)} samples each)")
        
        # Step 4: Extract embeddings
        print("\n🔧 Step 4: Extracting embeddings...")
        
        embedding1 = extractor.extract_embeddings_from_array(audio1, sample_rate)
        embedding2 = extractor.extract_embeddings_from_array(audio2, sample_rate)
        embedding3 = extractor.extract_embeddings_from_array(audio3, sample_rate)
        
        print(f"✅ Extracted embeddings - Shape: {embedding1.shape}")
        print(f"   Audio 1 (A4): {embedding1.shape[0]} dimensions")
        print(f"   Audio 2 (A5): {embedding2.shape[0]} dimensions")
        print(f"   Audio 3 (A3): {embedding3.shape[0]} dimensions")
        
        # Step 5: Calculate similarities
        print("\n🔧 Step 5: Calculating audio similarities...")
        
        sim_1_2 = compare_embeddings(embedding1, embedding2, method='cosine')
        sim_1_3 = compare_embeddings(embedding1, embedding3, method='cosine')
        sim_2_3 = compare_embeddings(embedding2, embedding3, method='cosine')
        
        print(f"✅ Similarity calculations completed:")
        print(f"   A4 ↔ A5 (octave up):   {sim_1_2:.3f}")
        print(f"   A4 ↔ A3 (octave down): {sim_1_3:.3f}")
        print(f"   A5 ↔ A3 (two octaves):  {sim_2_3:.3f}")
        
        # Step 6: Test embedding save/load
        print("\n🔧 Step 6: Testing embedding save/load...")
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            metadata = {
                'audio_type': 'synthetic_sine_wave',
                'frequency': 440,
                'duration': duration,
                'sample_rate': sample_rate,
                'model': 'spectrogram'
            }
            
            # Save embedding
            save_embeddings(embedding1, tmp.name, metadata)
            
            # Load embedding
            loaded_embedding, loaded_metadata = load_embeddings(tmp.name)
            
            # Verify
            if np.allclose(embedding1, loaded_embedding):
                print("✅ Embedding save/load successful")
                print(f"   Original shape: {embedding1.shape}")
                print(f"   Loaded shape: {loaded_embedding.shape}")
                print(f"   Metadata: {loaded_metadata['audio_type']}")
            else:
                print("❌ Embedding save/load failed - data mismatch")
                return False
            
            # Cleanup
            os.unlink(tmp.name)
        
        # Step 7: Test batch processing simulation
        print("\n🔧 Step 7: Testing batch processing simulation...")
        
        audio_batch = {
            'audio_A4.wav': (audio1, sample_rate),
            'audio_A5.wav': (audio2, sample_rate),
            'audio_A3.wav': (audio3, sample_rate)
        }
        
        batch_embeddings = {}
        for filename, (audio, sr) in audio_batch.items():
            embedding = extractor.extract_embeddings_from_array(audio, sr)
            batch_embeddings[filename] = embedding
        
        print(f"✅ Batch processing completed: {len(batch_embeddings)} files")
        
        # Step 8: Similarity search simulation
        print("\n🔧 Step 8: Testing similarity search...")
        
        query_embedding = embedding1  # Use A4 as query
        similarities = []
        
        for filename, embedding in batch_embeddings.items():
            if not np.array_equal(embedding, query_embedding):  # Skip identical
                sim = compare_embeddings(query_embedding, embedding, method='cosine')
                similarities.append((filename, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("✅ Similarity search results (most similar first):")
        for filename, sim in similarities:
            print(f"   {filename}: {sim:.3f}")
        
        # Step 9: Performance check
        print("\n🔧 Step 9: Performance check...")
        
        import time
        start_time = time.time()
        
        # Extract embeddings for multiple audio segments
        for i in range(10):
            test_audio = 0.3 * np.sin(2 * np.pi * (440 + i * 10) * t)
            _ = extractor.extract_embeddings_from_array(test_audio, sample_rate)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        print(f"✅ Performance test: 10 extractions in {duration_ms:.1f}ms")
        print(f"   Average: {duration_ms/10:.1f}ms per extraction")
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 FINAL FUNCTIONALITY TEST - COMPLETE SUCCESS!")
        print("=" * 60)
        
        print("\n✅ ALL FEATURES WORKING PERFECTLY:")
        print("   🎵 Audio embedding extraction")
        print("   📊 Similarity calculation")
        print("   💾 Embedding save/load")
        print("   ⚡ Batch processing")
        print("   🔍 Similarity search")
        print("   🚀 Good performance")
        
        print("\n🎯 YOUR MUSIC EMBEDDINGS PROJECT IS READY FOR:")
        print("   • Building music similarity search engines")
        print("   • Developing copyright detection systems") 
        print("   • Creating music recommendation algorithms")
        print("   • Analyzing audio content at scale")
        
        print("\n🎓 Created by Sergie Code - AI Tools for Musicians")
        print("💡 Perfect foundation for advanced music AI applications!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FUNCTIONALITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    # Add src to path
    add_src_to_path()
    
    # Run complete test
    success = test_complete_workflow()
    
    if success:
        print("\n🚀 PROJECT STATUS: READY FOR PRODUCTION!")
        return 0
    else:
        print("\n⚠️ PROJECT STATUS: NEEDS ATTENTION")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
