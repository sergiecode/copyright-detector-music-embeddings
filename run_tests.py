"""
Test runner script for music embeddings project.

This script runs all tests and provides a comprehensive report
of the project's functionality and health.

Author: Sergie Code
Usage: python run_tests.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🎵 {title}")
    print("="*60)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n🔧 {title}")
    print("-" * 40)

def check_dependencies():
    """Check if required dependencies are installed."""
    print_section("Checking Dependencies")
    
    required_packages = [
        'pytest', 'numpy', 'librosa', 'soundfile', 
        'scipy', 'pandas', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required dependencies are installed!")
    return True

def setup_test_environment():
    """Set up the test environment."""
    print_section("Setting Up Test Environment")
    
    # Add src to Python path
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    
    # Create test data directory if needed
    test_data_dir = project_root / 'test_data'
    test_data_dir.mkdir(exist_ok=True)
    print(f"✅ Test data directory ready: {test_data_dir}")
    
    return True

def run_specific_tests(test_type: str):
    """Run specific type of tests."""
    test_files = {
        'utils': 'tests/test_utils.py',
        'embeddings': 'tests/test_embeddings.py', 
        'integration': 'tests/test_integration.py'
    }
    
    if test_type not in test_files:
        print(f"❌ Unknown test type: {test_type}")
        return False
    
    print_section(f"Running {test_type.title()} Tests")
    
    cmd = ['python', '-m', 'pytest', test_files[test_type], '-v']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {test_type.title()} tests PASSED")
            return True
        else:
            print(f"❌ {test_type.title()} tests FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_type.title()} tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running {test_type} tests: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print_section("Running All Tests")
    
    cmd = ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short']
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        end_time = time.time()
        
        duration = end_time - start_time
        
        print(f"📊 Test Duration: {duration:.2f} seconds")
        
        if result.returncode == 0:
            print("✅ ALL TESTS PASSED!")
            print("\n📋 Test Summary:")
            print(result.stdout.split('\n')[-3:-1])  # Get summary lines
            return True
        else:
            print("❌ SOME TESTS FAILED")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def check_test_coverage():
    """Check test coverage of the codebase."""
    print_section("Checking Test Coverage")
    
    try:
        # Try to run coverage if available
        cmd = ['python', '-m', 'pytest', '--cov=src', '--cov-report=term-missing', 'tests/']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("📊 Coverage Report:")
            print(result.stdout)
            return True
        else:
            print("⚠️ Coverage analysis failed (coverage not installed?)")
            print("Install with: pip install pytest-cov")
            return False
            
    except FileNotFoundError:
        print("⚠️ Coverage tool not available")
        print("Install with: pip install pytest-cov")
        return False
    except Exception as e:
        print(f"❌ Error checking coverage: {e}")
        return False

def test_project_functionality():
    """Test basic project functionality without pytest."""
    print_section("Testing Basic Functionality")
    
    try:
        # Test imports
        from embeddings import AudioEmbeddingExtractor, compare_embeddings
        from utils import load_audio, save_embeddings
        print("✅ Module imports successful")
        
        # Test extractor initialization
        extractor = AudioEmbeddingExtractor(model_name='spectrogram')
        print("✅ Extractor initialization successful")
        
        # Test with synthetic audio
        import numpy as np
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        embedding = extractor.extract_embeddings_from_array(test_audio, 22050)
        print(f"✅ Embedding extraction successful - shape: {embedding.shape}")
        
        # Test embedding comparison
        embedding2 = extractor.extract_embeddings_from_array(test_audio * 0.5, 22050)
        similarity = compare_embeddings(embedding, embedding2, method='cosine')
        print(f"✅ Similarity calculation successful - similarity: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report."""
    print_header("MUSIC EMBEDDINGS PROJECT - TEST REPORT")
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"By: Sergie Code - AI Tools for Musicians")
    
    results = {
        'dependencies': False,
        'environment': False,
        'basic_functionality': False,
        'utils_tests': False,
        'embeddings_tests': False,
        'integration_tests': False,
        'all_tests': False
    }
    
    # Run all checks
    results['dependencies'] = check_dependencies()
    
    if results['dependencies']:
        results['environment'] = setup_test_environment()
        
        if results['environment']:
            results['basic_functionality'] = test_project_functionality()
            results['utils_tests'] = run_specific_tests('utils')
            results['embeddings_tests'] = run_specific_tests('embeddings')
            results['integration_tests'] = run_specific_tests('integration')
            results['all_tests'] = run_all_tests()
            
            # Try coverage check
            check_test_coverage()
    
    # Print final report
    print_header("FINAL TEST REPORT")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\n📊 Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 CONGRATULATIONS! All tests passed!")
        print("🚀 Your music embeddings project is working perfectly!")
        print("🎵 Ready for production use and further development!")
    elif passed >= total * 0.8:
        print("\n✅ Great job! Most tests passed.")
        print("🔧 Address the failing tests to achieve perfection.")
    elif passed >= total * 0.5:
        print("\n⚠️ Some issues detected.")
        print("🛠️ Review and fix the failing components.")
    else:
        print("\n❌ Major issues detected.")
        print("🆘 Significant debugging required.")
    
    print("\n" + "="*60)
    print("🎓 Created by Sergie Code - AI Tools for Musicians")
    print("💡 Subscribe to the YouTube channel for more AI tutorials!")
    print("="*60)

def main():
    """Main function."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type in ['utils', 'embeddings', 'integration']:
            setup_test_environment()
            run_specific_tests(test_type)
        elif test_type == 'coverage':
            setup_test_environment()
            check_test_coverage()
        elif test_type == 'basic':
            setup_test_environment()
            test_project_functionality()
        else:
            print("Usage: python run_tests.py [utils|embeddings|integration|coverage|basic]")
    else:
        generate_test_report()

if __name__ == "__main__":
    main()
