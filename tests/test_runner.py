"""
Comprehensive test runner for Assignment Omni.
Runs all test cases for API handling, LLM processing, and retrieval logic.
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from assignment_omni.config.settings import Settings


class TestRunner:
    """Main test runner class for Assignment Omni."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self):
        """Run all test suites."""
        print("=" * 60)
        print("ASSIGNMENT OMNI - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = datetime.now()
        
        # Test suites to run
        test_suites = [
            ("API Handling Tests", "tests/test_api_handling.py"),
            ("LLM Processing Tests", "tests/test_llm_processing.py"),
            ("Retrieval Logic Tests", "tests/test_retrieval_logic.py"),
            ("Weather Client Tests", "tests/test_weather_client.py")
        ]
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for suite_name, test_file in test_suites:
            print(f"\n{'='*20} {suite_name} {'='*20}")
            
            if not Path(test_file).exists():
                print(f"❌ Test file not found: {test_file}")
                continue
            
            # Run individual test suite
            suite_results = self.run_test_suite(test_file)
            
            # Update totals
            suite_tests = suite_results.get('total', 0)
            suite_passed = suite_results.get('passed', 0)
            suite_failed = suite_results.get('failed', 0)
            
            total_tests += suite_tests
            total_passed += suite_passed
            total_failed += suite_failed
            
            # Store results
            self.test_results[suite_name] = suite_results
            
            # Print suite summary
            print(f"\n{suite_name} Summary:")
            print(f"  Total: {suite_tests}")
            print(f"  Passed: {suite_passed}")
            print(f"  Failed: {suite_failed}")
            print(f"  Success Rate: {(suite_passed/suite_tests*100) if suite_tests > 0 else 0:.1f}%")
        
        self.end_time = datetime.now()
        self.print_final_summary(total_tests, total_passed, total_failed)
        
        return self.test_results
    
    def run_test_suite(self, test_file):
        """Run a specific test suite."""
        try:
            # Run pytest on the test file
            result = pytest.main([
                test_file,
                "-v",  # verbose
                "--tb=short",  # short traceback
                "--no-header",  # no header
                "--disable-warnings"  # disable warnings
            ])
            
            # Parse results (simplified)
            if result == 0:
                return {"total": 1, "passed": 1, "failed": 0, "status": "PASSED"}
            else:
                return {"total": 1, "passed": 0, "failed": 1, "status": "FAILED"}
                
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            return {"total": 0, "passed": 0, "failed": 1, "status": "ERROR"}
    
    def print_final_summary(self, total_tests, total_passed, total_failed):
        """Print final test summary."""
        duration = (self.end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("FINAL TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Completed at: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if total_failed == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"\n⚠️  {total_failed} TESTS FAILED")
        
        print("=" * 60)
    
    def run_specific_tests(self, test_types=None):
        """Run specific types of tests."""
        if test_types is None:
            test_types = ["api", "llm", "retrieval"]
        
        print("=" * 60)
        print("RUNNING SPECIFIC TEST TYPES")
        print("=" * 60)
        
        test_mapping = {
            "api": "tests/test_api_handling.py",
            "llm": "tests/test_llm_processing.py",
            "retrieval": "tests/test_retrieval_logic.py",
            "weather": "tests/test_weather_client.py"
        }
        
        for test_type in test_types:
            if test_type in test_mapping:
                test_file = test_mapping[test_type]
                print(f"\nRunning {test_type.upper()} tests...")
                self.run_test_suite(test_file)
    
    def check_environment(self):
        """Check if environment is properly set up for testing."""
        print("=" * 60)
        print("ENVIRONMENT CHECK")
        print("=" * 60)
        
        # Check if PDF file exists
        pdf_path = Path("EMROPUB_2019_en_23536.pdf")
        if pdf_path.exists():
            print("✅ PDF file found for RAG testing")
        else:
            print("❌ PDF file not found - RAG tests may fail")
        
        # Check environment variables
        settings = Settings.load()
        
        if settings.weather.openweather_api_key:
            print("✅ OpenWeather API key configured")
        else:
            print("⚠️  OpenWeather API key not configured - weather tests may fail")
        
        if settings.langsmith.api_key:
            print("✅ LangSmith API key configured")
        else:
            print("⚠️  LangSmith API key not configured - evaluation tests may fail")
        
        if settings.qdrant.url:
            print(f"✅ Qdrant URL configured: {settings.qdrant.url}")
        else:
            print("⚠️  Qdrant URL not configured - vector store tests may fail")
        
        print("=" * 60)


def main():
    """Main function to run tests."""
    runner = TestRunner()
    
    # Check environment first
    runner.check_environment()
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Return exit code based on results
    total_failed = sum(suite.get('failed', 0) for suite in results.values())
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
