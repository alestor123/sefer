"""
Main Application with latex.build_pdf PDF Generation
"""
import os
import sys
from question import QuestionExtractor
from ollama_solver import OllamaDeepSeekSolver

def main():
    print("=" * 70)
    print("🎯 AI QUESTION BANK GENERATOR WITH LATEX.BUILD_PDF")
    print("📚 PDF → Questions → Ollama AI → LaTeX → PDF")
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
    
    # STEP 2: Solve with Ollama + Convert to PDF
    print(f"\n🧠 STEP 2: SOLVING WITH OLLAMA + PDF GENERATION")
    print("-" * 40)
    
    solver = OllamaDeepSeekSolver(
        images_dir='temp',
        latex_dir='latex_output',
        model_name='llama3.1:8b'
    )
    
    workflow_results = solver.run_complete_workflow()
    
    if workflow_results:
        print("🎉 SUCCESS!")
        print(f"📊 Questions processed: {workflow_results['total_questions']}")
        print(f"📄 Final PDF: {workflow_results['pdf_path']}")
        print(f"📁 LaTeX files: {workflow_results['latex_dir']}/")
        print("=" * 70)
    else:
        print("❌ Workflow failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
