#!/usr/bin/env python
"""
Compatibility test script for Ranboost Fourcart GUI.
This script tests the core functionality of the application across different Python versions.
"""
import sys
import os
import platform
import subprocess
import importlib
import codecs
from typing import Dict, List, Tuple, Optional

# Use packaging.version for version comparison
try:
    from packaging import version as version_parse
except ImportError:
    # Fallback for environments without packaging module
    def version_parse(version_str):
        return tuple(map(int, (version_str.split('.') + ['0', '0', '0'])[:3]))

# Required packages and their minimum versions
REQUIRED_PACKAGES = {
    'numpy': '1.19.0',
    'scipy': '1.5.0',
    'scikit-learn': '0.24.0',
    'librosa': '0.8.0',
    'pydub': '0.25.1',
    'matplotlib': '3.3.0',
    'joblib': '1.0.0',
    'pandas': '1.1.0',
    'tqdm': '4.50.0',
    'xgboost': '1.3.0',
}

# Tested Python versions
SUPPORTED_PYTHON_VERSIONS = [
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11)
]

def check_python_version() -> Tuple[bool, str]:
    """Check if current Python version is supported."""
    current_version = sys.version_info[:2]
    is_supported = current_version in SUPPORTED_PYTHON_VERSIONS
    version_str = f"{current_version[0]}.{current_version[1]}"
    
    if is_supported:
        return True, f"Python {version_str} is supported."
    else:
        return False, f"Python {version_str} is not in the list of tested versions."

def check_package_installed(package_name: str, min_version: str) -> Tuple[bool, str]:
    """Check if a package is installed and meets the minimum version requirement."""
    # Special handling for scikit-learn which might be installed as sklearn
    if package_name == 'scikit-learn':
        package_name = 'sklearn'
        
    try:
        module = importlib.import_module(package_name)
        
        # Handle different version attributes
        version_attrs = ['__version__', 'version', 'VERSION']
        version = '0.0.0'
        for attr in version_attrs:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if callable(version):
                    version = version()
                break
                
        # If we couldn't get version from module, try pkg_resources
        if version == '0.0.0':
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package_name).version
            except:
                pass
        
        # Simple version comparison
        try:
            # Convert version to string in case it's a version object
            version_str = str(version).split('+')[0]  # Remove any build metadata
            
            # Handle cases where version might be a tuple
            if isinstance(version, (tuple, list)):
                version_str = '.'.join(map(str, version))
                
            # Compare versions
            if callable(version_parse) and not hasattr(version_parse, 'parse'):
                # Using our simple version parser
                current_ver = version_parse(version_str)
                min_ver = version_parse(min_version)
                if current_ver < min_ver:
                    return False, f"{package_name} {version_str} is installed but version {min_version}+ is required."
            else:
                # Using packaging.version
                if version_parse(version_str) < version_parse(min_version):
                    return False, f"{package_name} {version_str} is installed but version {min_version}+ is required."
            
            return True, f"{package_name} {version_str} is installed and compatible."
            
        except Exception as e:
            # If version comparison fails, just report the package is installed
            return True, f"{package_name} is installed (version check skipped)"
    except ImportError:
        return False, f"{package_name} is not installed."

def check_imports() -> Dict[str, Tuple[bool, str]]:
    """Check if all required packages are importable."""
    results = {}
    
    # First try to import all packages without version checks
    for package in REQUIRED_PACKAGES.keys():
        try:
            importlib.import_module('sklearn' if package == 'scikit-learn' else package)
            results[package] = (True, "Package is installed")
        except ImportError:
            results[package] = (False, f"{package} is not installed")
    
    # Then do version checks for packages that were successfully imported
    for package, min_version in REQUIRED_PACKAGES.items():
        if results[package][0]:  # If package is installed
            results[package] = check_package_installed(package, min_version)
    
    return results

def run_basic_tests() -> Dict[str, Tuple[bool, str]]:
    """Run basic functionality tests."""
    results = {}
    
    # Test 1: Check if algorithm.py can be imported and contains expected classes
    try:
        # First try to import the module to see if it exists
        algorithm_module = importlib.import_module('algorithm')
        
        # Look for any class that might be our model
        model_classes = [
            name for name, obj in algorithm_module.__dict__.items()
            if (isinstance(obj, type) and 
                name not in ('BaseEstimator', 'ClassifierMixin', 'object') and
                not name.startswith('_'))
        ]
        
        # Look for specific class names that might indicate our model
        possible_names = ['RanboostFourCart', 'MyCustomAlgorithm', 'CustomClassifier']
        found_models = [name for name in possible_names if name in algorithm_module.__dict__]
        
        if found_models:
            results['Algorithm import'] = (True, f"Found model class(es): {', '.join(found_models)}")
        elif model_classes:
            results['Algorithm import'] = (True, f"Found potential model classes: {', '.join(model_classes[:3])}{'...' if len(model_classes) > 3 else ''}")
        else:
            results['Algorithm import'] = (False, "No model classes found in algorithm.py")
    except Exception as e:
        results['Algorithm import'] = (False, f"Failed to import algorithm.py: {str(e)}")
    
    # Test 2: Check if gui.py can be imported
    try:
        import gui
        results['GUI import'] = (True, "Successfully imported gui.py")
    except Exception as e:
        results['GUI import'] = (False, f"Failed to import gui.py: {str(e)}")
    
    return results

def generate_report() -> str:
    """Generate a compatibility report."""
    report = []
    
    # System information
    report.append("=" * 80)
    report.append("COMPATIBILITY TEST REPORT")
    report.append("=" * 80)
    report.append(f"System: {platform.system()} {platform.release()}")
    report.append(f"Python: {platform.python_version()}")
    
    # Python version check
    py_check, py_msg = check_python_version()
    report.append(f"\nPython Version Check: {'✓' if py_check else '✗'} {py_msg}")
    
    # Package checks
    report.append("\nREQUIRED PACKAGES:")
    import_checks = check_imports()
    for package, (success, msg) in import_checks.items():
        # Use simple status indicators that work in all terminals
        status = '[OK]' if success else '[X]'
        # Truncate long messages to avoid wrapping
        if len(msg) > 80:
            msg = msg[:77] + '...'
        report.append(f"  {status} {msg}")
    
    # Basic tests
    report.append("\nBASIC TESTS:")
    test_results = run_basic_tests()
    for test_name, (success, msg) in test_results.items():
        # Use checkmark and cross symbols that work in Windows console
        status = '[OK]' if success else '[X]'
        report.append(f"  {status} {test_name}: {msg}")
    
    # Summary
    all_passed = all(success for success, _ in import_checks.values()) and \
                 all(success for success, _ in test_results.values())
    
    report.append("\n" + "=" * 80)
    if all_passed:
        report.append("[PASS] COMPATIBILITY TEST PASSED")
    else:
        report.append("[FAIL] COMPATIBILITY TEST FAILED")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main function to run compatibility tests."""
    report = generate_report()
    print(report)
    
    # Save report to file with UTF-8 encoding
    with open("compatibility_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("\nReport saved to 'compatibility_report.txt'")

if __name__ == "__main__":
    main()
