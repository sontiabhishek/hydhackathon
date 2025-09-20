# This script ensures all dependencies are installed before launching the Streamlit app.
# It provides a robust, automated way to set up the environment, preventing runtime errors.

import sys
import subprocess

try:
    # Use the modern importlib.metadata for checking installed packages (Python 3.8+)
    from importlib.metadata import distributions, PackageNotFoundError
    from packaging.requirements import Requirement
except ImportError:
    # Fallback to the deprecated pkg_resources for older Python versions
    # Note: This requires setuptools to be installed.
    print("Warning: Using deprecated 'pkg_resources'. Consider upgrading to Python 3.8+.")
    try:
        from pkg_resources import get_distribution, DistributionNotFound as PackageNotFoundError
        # Create a compatible 'Requirement' class for parsing
        class Requirement:
            def __init__(self, req_string):
                parts = req_string.strip().split('==')
                self.name = parts[0]
    except ImportError:
        print("Error: 'setuptools' is required for this script to run on Python < 3.8. Please install it.")
        sys.exit(1)

def get_installed_packages():
    """Returns a set of installed package names."""
    try:
        return {dist.metadata['name'].lower() for dist in distributions()}
    except NameError:
        try:
            from pip._vendor import pkg_resources
            return {pkg.key for pkg in pkg_resources.working_set}
        except (ImportError, NameError):
             return {}


def check_and_install_dependencies():
    """
    Reads requirements.txt, checks if dependencies are met, and installs them if not.
    """
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print("Checking application dependencies...")
        installed_packages = get_installed_packages()

        # Extract just the package name from the requirements file (e.g., 'scikit-learn' from 'scikit-learn==1.4.2')
        required_packages = {req.split('==')[0].lower() for req in requirements}

        missing_reqs = required_packages - installed_packages

        if not missing_reqs:
            print("✅ All dependencies are satisfied.")
            return

        print(f"🟡 Found missing dependencies: {', '.join(missing_reqs)}. Installing...")
        # Use sys.executable to ensure we use the pip from the current virtual environment.
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully.")

    except FileNotFoundError:
        print("Error: 'requirements.txt' not found. Cannot check dependencies.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during dependency installation: {e}")
        sys.exit(1)

def launch_app():
    """
    Launches the Streamlit application.
    """
    print("\nLaunching the Streamlit application...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    check_and_install_dependencies()
    launch_app()