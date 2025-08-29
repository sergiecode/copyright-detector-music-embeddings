"""
Dependency installer for music embeddings project.

This script installs all required dependencies with proper version management
to ensure compatibility across different systems.

Author: Sergie Code
"""

import subprocess
import sys
import os
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

def run_command(command: list, description: str):
    """Run a command and return success status."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print_section("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def upgrade_pip():
    """Upgrade pip to latest version."""
    print_section("Upgrading pip")
    
    return run_command([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip"
    ], "pip upgrade")

def install_core_dependencies():
    """Install core dependencies with proper versions."""
    print_section("Installing Core Dependencies")
    
    # Core packages with specific versions for compatibility
    core_packages = [
        "numpy>=1.21.0,<2.0.0",  # Fix numpy version for librosa compatibility
        "scipy>=1.7.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    for package in core_packages:
        success = run_command([
            sys.executable, "-m", "pip", "install", package
        ], f"Installing {package}")
        
        if not success:
            print(f"⚠️ Failed to install {package}")
            return False
    
    return True

def install_audio_dependencies():
    """Install audio processing dependencies."""
    print_section("Installing Audio Processing Dependencies")
    
    audio_packages = [
        "librosa>=0.10.0",
        "soundfile>=0.10.0",
        "audioread>=2.1.9"
    ]
    
    for package in audio_packages:
        success = run_command([
            sys.executable, "-m", "pip", "install", package
        ], f"Installing {package}")
        
        if not success:
            print(f"⚠️ Failed to install {package}")
            return False
    
    return True

def install_ml_dependencies():
    """Install machine learning dependencies."""
    print_section("Installing ML Dependencies")
    
    # Try to install TensorFlow and PyTorch
    ml_packages = [
        ("tensorflow>=2.10.0", "TensorFlow"),
        ("torch>=1.12.0", "PyTorch"),
        ("torchaudio>=0.12.0", "TorchAudio")
    ]
    
    success_count = 0
    for package, name in ml_packages:
        success = run_command([
            sys.executable, "-m", "pip", "install", package
        ], f"Installing {name}")
        
        if success:
            success_count += 1
        else:
            print(f"⚠️ {name} installation failed (optional)")
    
    return success_count > 0

def install_optional_dependencies():
    """Install optional dependencies for enhanced functionality."""
    print_section("Installing Optional Dependencies")
    
    optional_packages = [
        ("openl3>=0.4.1", "OpenL3"),
        ("transformers>=4.21.0", "Transformers (for AudioCLIP)"),
        ("jupyter>=1.0.0", "Jupyter Notebook"),
        ("pytest>=7.1.0", "pytest (for testing)"),
        ("ipywidgets>=7.7.0", "IPython widgets")
    ]
    
    installed_count = 0
    for package, name in optional_packages:
        success = run_command([
            sys.executable, "-m", "pip", "install", package
        ], f"Installing {name}")
        
        if success:
            installed_count += 1
        else:
            print(f"⚠️ {name} installation failed (optional)")
    
    print(f"📊 Installed {installed_count}/{len(optional_packages)} optional packages")
    return True

def install_from_requirements():
    """Install from requirements.txt if available."""
    print_section("Installing from requirements.txt")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("⚠️ requirements.txt not found")
        return False
    
    # First, fix numpy version in requirements
    with open(requirements_file, 'r') as f:
        content = f.read()
    
    # Replace numpy version to be compatible
    content = content.replace("numpy>=1.21.0", "numpy>=1.21.0,<2.0.0")
    
    with open(requirements_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed numpy version in requirements.txt")
    
    return run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], "Installing from requirements.txt")

def verify_installation():
    """Verify that key packages are installed correctly."""
    print_section("Verifying Installation")
    
    test_imports = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile")
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"❌ {name} - Failed to import")
    
    # Test optional imports
    optional_imports = [
        ("openl3", "OpenL3"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow")
    ]
    
    optional_count = 0
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name} (optional)")
            optional_count += 1
        except ImportError:
            print(f"⚠️ {name} (optional) - Not available")
    
    print(f"\n📊 Core packages: {success_count}/{len(test_imports)}")
    print(f"📊 Optional packages: {optional_count}/{len(optional_imports)}")
    
    return success_count >= len(test_imports) - 1  # Allow one failure

def main():
    """Main installation function."""
    print_header("MUSIC EMBEDDINGS PROJECT - DEPENDENCY INSTALLER")
    print("By: Sergie Code - AI Tools for Musicians")
    print("This script will install all required dependencies for the project.")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Python version incompatible. Please upgrade Python.")
        return False
    
    # Run installation steps
    steps = [
        ("Upgrade pip", upgrade_pip),
        ("Install core dependencies", install_core_dependencies),
        ("Install audio dependencies", install_audio_dependencies),
        ("Install ML dependencies", install_ml_dependencies),
        ("Install optional dependencies", install_optional_dependencies),
        ("Verify installation", verify_installation)
    ]
    
    results = {}
    for step_name, step_func in steps:
        results[step_name] = step_func()
    
    # Summary
    print_header("INSTALLATION SUMMARY")
    
    success_count = sum(results.values())
    total_steps = len(results)
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{step_name:<30} {status}")
    
    print(f"\n📊 Overall: {success_count}/{total_steps} steps successful")
    
    if success_count >= total_steps - 1:  # Allow one optional failure
        print("\n🎉 INSTALLATION SUCCESSFUL!")
        print("🚀 Your music embeddings project is ready to use!")
        
        print("\n📋 Next Steps:")
        print("1. Run verification: python verify_project.py")
        print("2. Run tests: python run_tests.py")
        print("3. Start Jupyter: jupyter notebook notebooks/embedding_demo.ipynb")
        
    else:
        print("\n⚠️ INSTALLATION ISSUES DETECTED")
        print("🛠️ Please review the errors above and try manual installation.")
        print("💡 You can still use basic functionality with core packages.")
    
    print("\n" + "="*60)
    print("🎓 Created by Sergie Code - AI Tools for Musicians")
    print("💡 Subscribe to the YouTube channel for more AI tutorials!")
    print("="*60)
    
    return success_count >= total_steps - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
