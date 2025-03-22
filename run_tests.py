#!/usr/bin/env python3
"""
NBA Sentiment Predictor - Test Runner

This script runs all test files for the NBA Sentiment Predictor.
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_tests')

def run_test_file(test_file: str) -> bool:
    """
    Run a specific test file and return whether it succeeded.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        bool: True if test succeeded, False otherwise
    """
    logger.info(f"Running test file: {test_file}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    # Print the output
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(result.stderr)
    
    success = result.returncode == 0
    status = "PASSED" if success else "FAILED"
    logger.info(f"Test {status} in {elapsed_time:.2f} seconds")
    
    return success

def run_all_tests():
    """Run all test files and report results."""
    logger.info(f"=" * 60)
    logger.info(f"NBA Sentiment Predictor Tests - {datetime.now()}")
    logger.info(f"=" * 60)
    
    # List of test files to run
    test_files = [
        "test_database.py",
        "test_nba_scraping.py"
    ]
    
    # Check if test files exist
    for test_file in test_files[:]:
        if not os.path.exists(test_file):
            logger.error(f"Test file not found: {test_file}")
            test_files.remove(test_file)
    
    if not test_files:
        logger.error("No test files found. Exiting.")
        return False
    
    # Run all test files
    results = {}
    for test_file in test_files:
        results[test_file] = run_test_file(test_file)
    
    # Print summary
    logger.info(f"=" * 60)
    logger.info("TEST SUMMARY")
    logger.info(f"=" * 60)
    
    for test_file, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_file}: {status}")
    
    # Overall result
    overall_success = all(results.values())
    overall_status = "PASSED" if overall_success else "FAILED"
    logger.info(f"=" * 60)
    logger.info(f"OVERALL RESULT: {overall_status}")
    logger.info(f"=" * 60)
    
    return overall_success

def main():
    """Main function."""
    try:
        success = run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())