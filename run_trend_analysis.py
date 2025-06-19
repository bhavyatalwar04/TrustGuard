
"""
Standalone Trend Analysis Runner

This script can be run directly to perform trend analysis on your preprocessed data.
Place this file in your project root directory (TruthGuard/) and run it.

Usage:
    python run_trend_analysis.py
    
The script will automatically find the most recent preprocessed file matching the pattern:
preprocessed_reddit_posts_advanced_*.csv
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path so we can import our modules
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now we can import our trend detection module
try:
    from src.processing.trend_engine.trend_detector import TrendDetectionEngine, print_analysis_results
    print("✅ Successfully imported TrendDetectionEngine")
except ImportError as e:
    print(f"❌ Failed to import TrendDetectionEngine: {e}")
    print("Make sure you have all required packages installed:")
    print("pip install pandas gensim scikit-learn numpy")
    sys.exit(1)

def main():
    """Main function to run trend analysis."""
    print("\n🎯 SOCIAL MEDIA TREND ANALYSIS")
    print("=" * 50)
    
    # Setup data directories
    data_dir = project_root / "data" / "processed_data"
    
    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Looking for data in: {data_dir}")
    
    try:
        # Initialize trend detector
        trend_detector = TrendDetectionEngine(data_dir=str(data_dir))
        
        # Run full analysis
        print("\n🔍 Starting analysis...")
        results = trend_detector.run_full_analysis()
        
        if results:
            # Print results
            print_analysis_results(results)
            print(f"\n🎉 Analysis completed successfully!")
            
            # Show output file location
            if 'output_file' in results:
                print(f"📄 Enhanced data saved to: {results['output_file']}")
                
        else:
            print("❌ No results generated. Check if you have preprocessed data files.")
            print(f"Expected files matching pattern: preprocessed_reddit_posts_advanced_*.csv")
            print(f"In directory: {data_dir}")
            
            # List files in the directory for debugging
            if data_dir.exists():
                files = list(data_dir.glob("*.csv"))
                if files:
                    print(f"\nFound {len(files)} CSV files:")
                    for file in files:
                        print(f"  - {file.name}")
                else:
                    print(f"\nNo CSV files found in {data_dir}")
            else:
                print(f"\nDirectory {data_dir} does not exist.")
                
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("\nMake sure you have preprocessed data files in the correct location.")
        print("Expected file pattern: preprocessed_reddit_posts_advanced_YYYYMMDD_HHMMSS.csv")
        print(f"Expected location: {data_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()