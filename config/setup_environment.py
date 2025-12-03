#!/usr/bin/env python3
"""
CCTV Anomaly Detection System - Environment Setup

This script sets up the Python environment and installs required dependencies.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print setup banner."""
    print("🔧 CCTV Anomaly Detection System - Environment Setup")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("💡 Please install Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install Python requirements."""
    print("📦 Installing Python packages...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Try to upgrade pip (optional)
        print("⬆️ Upgrading pip...")
        pip_upgrade = subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                                   capture_output=True, text=True)
        if pip_upgrade.returncode == 0:
            print("✅ Pip upgraded successfully")
        else:
            print("⚠️ Pip upgrade failed, continuing with current version...")
        
        # Install requirements
        print("📥 Installing requirements...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully!")
            return True
        else:
            print("❌ Failed to install some packages:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
            # Try with --user flag
            print("🔄 Trying with --user flag...")
            user_result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '-r', 'requirements.txt'], 
                                       capture_output=True, text=True)
            
            if user_result.returncode == 0:
                print("✅ Packages installed with --user flag!")
                return True
            else:
                print("❌ Installation failed even with --user flag")
                return False
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def setup_config():
    """Setup configuration files."""
    print("⚙️ Setting up configuration...")
    
    # Create config from template if needed
    if not os.path.exists('config/config.yaml') and os.path.exists('config/config.yaml.template'):
        print("📝 Creating config.yaml from template...")
        subprocess.run(['cp', 'config/config.yaml.template', 'config/config.yaml'])
        print("✅ Configuration file created")
    
    # Create necessary directories
    directories = ['logs', 'storage', 'temp', 'backups']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def check_optional_dependencies():
    """Check for optional dependencies."""
    print("🔍 Checking optional dependencies...")
    
    # Check for GPU support
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA GPU support available")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple Metal GPU support available")
        else:
            print("⚠️ No GPU acceleration available (CPU only)")
    except ImportError:
        print("❌ PyTorch not installed")
    
    # Check for Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker available")
        else:
            print("⚠️ Docker not available")
    except FileNotFoundError:
        print("⚠️ Docker not installed")

def verify_installation():
    """Verify the installation by importing key modules."""
    print("🧪 Verifying installation...")
    
    test_imports = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('sklearn', 'Scikit-learn'),
        ('flask', 'Flask'),
        ('yaml', 'PyYAML')
    ]
    
    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} import failed")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"❌ Some imports failed: {failed_imports}")
        return False
    
    print("✅ All imports successful!")
    return True

def show_next_steps():
    """Show next steps after setup."""
    print("\n🎯 Setup Complete! Next Steps:")
    print("=" * 40)
    print("1. Run the system:")
    print("   ./start.sh")
    print()
    print("2. Or use direct commands:")
    print("   python3 run_project.py --mode realtime --stream 0")
    print("   python3 run_project.py --mode status")
    print()
    print("3. For help:")
    print("   python3 run_project.py --help")
    print()
    print("Happy detecting! 🛡️")

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Setup configuration
    setup_config()
    
    # Check optional dependencies
    check_optional_dependencies()
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup completed but some imports failed")
        print("💡 You may need to install additional system dependencies")
    
    # Show next steps
    show_next_steps()

if __name__ == '__main__':
    main()