import os
import librosa
import numpy as np
from algorithm import extract_features

def test_feature_extraction(audio_file):
    print("Testing feature extraction...")
    features = extract_features(audio_file, debug=True)
    print("\nFeature extraction test complete!")
    print(f"Total features extracted: {len(features)}")

if __name__ == "__main__":
    # Test with a sample audio file
    test_file = "test/test.mp3"  # Update this path if needed
    if os.path.exists(test_file):
        test_feature_extraction(test_file)
    else:
        print(f"Test file not found: {test_file}")
        print("Please update the test file path in the script.")
