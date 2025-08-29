"""
Quick verification script for music embeddings project.

This script performs basic checks without heavy dependencies
to verify the project structure and core functionality.

Author: Sergie Code
"""

import sys
import os
import numpy as np
from pathlib import Path

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*50)
    print(f"🎵 {title}")
    print("="*50)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n🔧 {title}")
    print("-" * 30)

def check_project_structure():
    """Check if project structure is correct."""
    print_section("Checking Project Structure")
    
    project_root = Path(".")
    expected_files = [
        "src/__init__.py",
        "src/embeddings.py", 
        "src/utils.py",
        "tests/__init__.py",
        "tests/test_utils.py",
        "tests/test_embeddings.py",
        "tests/test_integration.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "run_tests.py",
        "pytest.ini"
    ]
    
    all_present = True
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

def check_basic_imports():
    """Check if basic imports work."""
    print_section("Checking Basic Imports")
    
    # Add src to path
    src_path = Path(".") / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        # Test basic Python imports
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
        
        # Test custom module imports
        import embeddings
        import utils
        print("✅ Custom modules (embeddings, utils)")
        
        # Test class availability
        from embeddings import AudioEmbeddingExtractor, compare_embeddings
        from utils import get_supported_formats
        print("✅ Main classes and functions")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_simple_functionality():
    """Test simple functionality without heavy audio processing."""
    print_section("Testing Simple Functionality")
    
    try:
        # Test supported formats
        from utils import get_supported_formats
        formats = get_supported_formats()
        print(f"✅ Supported formats: {len(formats)} formats")
        
        # Test embedding comparison with dummy data
        from embeddings import compare_embeddings
        emb1 = np.random.randn(100)
        emb2 = np.random.randn(100)
        
        similarity = compare_embeddings(emb1, emb2, method='cosine')
        print(f"✅ Embedding comparison: {similarity:.3f}")
        
        # Test perfect similarity
        perfect_sim = compare_embeddings(emb1, emb1, method='cosine')
        assert abs(perfect_sim - 1.0) < 1e-10
        print(f"✅ Perfect similarity test: {perfect_sim:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def create_simple_test_audio():
    """Create simple test audio without librosa."""
    print_section("Creating Test Audio")
    
    try:
        # Create simple sine wave
        duration = 2.0
        sample_rate = 22050
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        print(f"✅ Created test audio: {len(audio)} samples at {sample_rate} Hz")
        return audio, sample_rate
        
    except Exception as e:
        print(f"❌ Test audio creation failed: {e}")
        return None, None

def test_without_heavy_deps():
    """Test functionality that doesn't require heavy dependencies."""
    print_section("Testing Core Features (No Heavy Dependencies)")
    
    try:
        from embeddings import AudioEmbeddingExtractor
        
        # This should fall back to simple spectrogram without librosa
        print("🔧 Initializing extractor...")
        
        # We'll test with minimal functionality
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        print("✅ Extractor initialized (may have fallback warnings)")
        
        return True
        
    except Exception as e:
        print(f"❌ Core features test failed: {e}")
        return False

def main():
    """Main verification function."""
    print_header("MUSIC EMBEDDINGS PROJECT - QUICK VERIFICATION")
    print("By: Sergie Code - AI Tools for Musicians")
    print("Purpose: Verify project setup and basic functionality")
    
    results = {}
    
    # Run checks
    results['structure'] = check_project_structure()
    results['imports'] = check_basic_imports()
    results['simple_functionality'] = test_simple_functionality()
    
    # Create test audio
    audio, sr = create_simple_test_audio()
    results['test_audio'] = audio is not None
    
    # Test core features (may show warnings)
    results['core_features'] = test_without_heavy_deps()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        if result:
            passed += 1
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    score = passed / total * 100
    print(f"\n📊 Verification Score: {passed}/{total} ({score:.1f}%)")
    
    if score >= 80:
        print("\n🎉 EXCELLENT! Project structure and core functionality verified!")
        print("🚀 Ready for full testing and development!")
        
        print("\n📋 Next Steps:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Run full tests: python run_tests.py")
        print("3. Try the Jupyter notebook: jupyter notebook notebooks/embedding_demo.ipynb")
        
    elif score >= 60:
        print("\n✅ GOOD! Basic structure is correct.")
        print("🔧 Address any missing components and run full tests.")
        
    else:
        print("\n⚠️ ISSUES DETECTED! Please review the project setup.")
        print("🛠️ Check missing files and dependencies.")
    
    print("\n" + "="*50)
    print("🎓 Created by Sergie Code")
    print("💡 AI Tools for Musicians")
    print("="*50)

if __name__ == "__main__":
    main()
