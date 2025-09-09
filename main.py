"""
Complete Integrated Workflow with Ollama
1. Extract questions using question.py  
2. Solve questions using Ollama + DeepSeek R1
3. Generate LaTeX question bank
4. Compile to final PDF
"""

import os
import sys
from question import QuestionExtractor
from ollama_solver import OllamaDeepSeekSolver

def main():
    print("=" * 70)
    print("🎯 AI QUESTION BANK GENERATOR WITH OLLAMA")
    print("📚 PDF → Questions → Ollama DeepSeek R1 → LaTeX → PDF")
    print("=" * 70)
    
    if len(sys.argv) != 2:
        print("Usage: python main.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # STEP 1: Extract Questions
    print("\n🔍 STEP 1: EXTRACTING QUESTIONS")
    print("-" * 40)
    
    extractor = QuestionExtractor()
    results = extractor.process_pdf(pdf_path)
    
    if not results["success"]:
        print(f"❌ Extraction failed: {results['message']}")
        sys.exit(1)
    
    print(f"✅ Extracted {results['questions_extracted']} questions")
    
    # STEP 2: Solve with Ollama + DeepSeek R1
    print("\n🧠 STEP 2: SOLVING WITH OLLAMA + DEEPSEEK R1")
    print("-" * 40)
    
    solver = OllamaDeepSeekSolver(
        images_dir='temp',
        latex_dir='latex_output',
        model_name='deepseek-r1:1.5b'  # Use your available model
    )
    
    workflow_results = solver.run_complete_workflow()
    
    if workflow_results:
        print(f"\n🎉 SUCCESS!")
        print(f"📊 Questions processed: {workflow_results['total_questions']}")
        print(f"📄 Final PDF: {workflow_results['pdf_path']}")
        print(f"📁 LaTeX source: {workflow_results['latex_dir']}")
        
        print(f"\n💡 Your complete question bank PDF is ready!")
        print(f"📖 Open: {workflow_results['pdf_path']}")
        
    else:
        print("❌ Workflow failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
