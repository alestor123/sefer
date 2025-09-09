"""
Main Application
PDF Question Extraction Tool
"""

import os
import sys
from question import QuestionExtractor

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🎯 PDF QUESTION EXTRACTOR")
    print("📚 Structure-aware question extraction from exam papers")
    print("=" * 60)

def print_results(results):
    """Print extraction results"""
    if results["success"]:
        print(f"✅ SUCCESS!")
        print(f"📊 Questions found: {results['questions_found']}")
        print(f"📸 Questions extracted: {results['questions_extracted']}")
        print(f"📁 Output directory: temp/")
        print(f"📋 Report: {os.path.basename(results['report_path'])}")
        print(f"🌐 Viewer: {os.path.basename(results['viewer_path'])}")
        print(f"\n💡 Open temp/viewer.html in your browser to view all questions")
    else:
        print(f"❌ FAILED: {results['message']}")

def main():
    print_banner()
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <pdf_file>")
        print("\nExample:")
        print("  python main.py exam_paper.pdf")
        print("  python main.py test.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Validate PDF file
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File '{pdf_path}' not found")
        sys.exit(1)
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"❌ Error: '{pdf_path}' is not a PDF file")
        sys.exit(1)
    
    print(f"📄 Processing: {pdf_path}")
    
    # Initialize extractor
    extractor = QuestionExtractor()
    
    # Check if temp directory is clean (optional cleanup)
    if os.path.exists("temp") and os.listdir("temp"):
        response = input("🗑️  Temp directory contains files. Clean it? (y/n): ")
        if response.lower() == 'y':
            extractor.clean_temp_directory()
            print("✅ Temp directory cleaned")
    
    # Process PDF
    print("🚀 Starting extraction...")
    results = extractor.process_pdf(pdf_path)
    
    # Print results
    print("\n" + "=" * 60)
    print_results(results)
    print("=" * 60)

if __name__ == "__main__":
    main()
