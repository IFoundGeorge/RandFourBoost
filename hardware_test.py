#!/usr/bin/env python
"""
Hardware Compatibility Test for Ranboost Fourcart
This script checks the system's hardware capabilities for running the application.
"""
import os
import sys
import platform
import psutil
import time
import subprocess
import multiprocessing
from typing import Dict, Tuple, List, Optional
import numpy as np

def get_system_info() -> Dict[str, str]:
    """Get basic system information."""
    return {
        "System": f"{platform.system()} {platform.release()}",
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "CPU Cores": str(multiprocessing.cpu_count()),
        "Total RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    }

def check_cpu_performance() -> Dict[str, float]:
    """Test CPU performance with a simple calculation."""
    start_time = time.time()
    # Perform a CPU-intensive calculation
    result = 0
    for i in range(10**7):
        result += i * i
    cpu_time = time.time() - start_time
    
    return {
        "CPU Test Time (lower is better)": cpu_time,
        "CPU Score": 1.0 / (cpu_time + 0.001)  # Simple score calculation
    }

def check_memory_usage() -> Dict[str, str]:
    """Check memory usage and availability."""
    vm = psutil.virtual_memory()
    return {
        "Total RAM": f"{vm.total / (1024**3):.2f} GB",
        "Available RAM": f"{vm.available / (1024**3):.2f} GB",
        "Used RAM": f"{vm.used / (1024**3):.2f} GB",
        "RAM Usage %": f"{vm.percent}%"
    }

def check_disk_space() -> Dict[str, str]:
    """Check disk space and performance."""
    try:
        disk = psutil.disk_usage('.')
        return {
            "Total Disk Space": f"{disk.total / (1024**3):.2f} GB",
            "Used Disk Space": f"{disk.used / (1024**3):.2f} GB",
            "Free Disk Space": f"{disk.free / (1024**3):.2f} GB",
            "Disk Usage %": f"{disk.percent}%"
        }
    except Exception as e:
        return {"Disk Check Error": str(e)}

def check_gpu() -> Dict[str, str]:
    """Check for GPU availability and basic info."""
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["GPU Available"] = "Yes (CUDA)"
            gpu_info["GPU Count"] = str(torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                gpu_info[f"GPU {i} Name"] = torch.cuda.get_device_name(i)
                gpu_info[f"GPU {i} Memory"] = f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB"
        else:
            gpu_info["GPU Available"] = "No CUDA-capable GPU found"
    except ImportError:
        gpu_info["GPU Check"] = "PyTorch not installed, cannot detect GPU"
    
    return gpu_info

def check_audio_devices() -> Dict[str, str]:
    """Check for audio device availability."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return {
            "Audio Devices": f"Found {len(devices)} audio devices",
            "Default Input Device": f"{sd.query_devices(kind='input')['name']}",
            "Default Output Device": f"{sd.query_devices(kind='output')['name']}"
        }
    except Exception as e:
        return {"Audio Check": f"Audio device check failed: {str(e)}"}

def run_hardware_test() -> Dict[str, Dict[str, str]]:
    """Run all hardware tests and return results."""
    results = {}
    
    print("Running hardware compatibility tests...\n")
    
    # System Information
    results["System Information"] = get_system_info()
    
    # CPU Test
    print("Testing CPU performance...")
    results["CPU Performance"] = check_cpu_performance()
    
    # Memory Test
    print("Checking memory usage...")
    results["Memory Usage"] = check_memory_usage()
    
    # Disk Test
    print("Checking disk space...")
    results["Disk Space"] = check_disk_space()
    
    # GPU Test
    print("Checking GPU...")
    results["GPU Information"] = check_gpu()
    
    # Audio Test
    print("Checking audio devices...")
    results["Audio Devices"] = check_audio_devices()
    
    return results

def print_results(results: Dict[str, Dict[str, str]]):
    """Print the test results in a readable format."""
    print("\n" + "="*80)
    print("HARDWARE COMPATIBILITY REPORT")
    print("="*80)
    
    for category, tests in results.items():
        print(f"\n{category}:")
        print("-" * (len(category) + 1))
        for test, result in tests.items():
            if isinstance(result, float):
                print(f"  {test}: {result:.4f}")
            else:
                print(f"  {test}: {result}")
    
    print("\n" + "="*80)
    print("HARDWARE TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    # Check for required packages
    required_packages = ['psutil']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"The following required packages are missing: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    # Run the tests
    results = run_hardware_test()
    print_results(results)
    
    # Save results to file
    with open("hardware_compatibility_report.txt", "w", encoding='utf-8') as f:
        import json
        json.dump(results, f, indent=4)
    
    print("\nReport saved to 'hardware_compatibility_report.txt'")
    print("Note: For GPU testing, install PyTorch (pip install torch torchvision torchaudio)")
    print("Note: For audio device testing, install sounddevice (pip install sounddevice)")
