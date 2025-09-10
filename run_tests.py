#!/usr/bin/env python3
"""
Simple script to run all tests for Assignment Omni.
This script runs the comprehensive test suite using the sample PDF.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run all tests for the assignment."""
    print("Assignment Omni - Running Test Suite")
    print("=" * 50)
    
    # Check if PDF file exists
    pdf_path = Path("EMROPUB_2019_en_23536.pdf")
    if not pdf_path.exists():
        print("❌ PDF file not found: EMROPUB_2019_en_23536.pdf")
        print("   Please ensure the PDF file is in the project root directory.")
        return 1
    
    print("✅ PDF file found for RAG testing")
    
    # Run the test runner
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--disable-warnings"
        ], capture_output=False)
        
        if result.returncode == 0:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print(f"\n⚠️  Some tests failed (exit code: {result.returncode})")
            return result.returncode
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
