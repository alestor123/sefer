"""
Main Application with latex2pdf PDF Generation
"""

import os
import sys
from question import QuestionExtractor
from ollama_solver import OllamaDeepSeekSolver

def main():
    print("=" * 70)
    print("🎯 AI QUESTION BANK GENERATOR WITH LATEX2PDF")
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
        model_name='deepseek-coder:latest'  # Use your available model
    )
    
    workflow_results = solver.run_complete_workflow()
    
    if workflow_results:
        print("🎉 SUCCESS!")
        print(f"📊 Questions processed: {workflow_results['total_questions']}")
        
        # Fix for missing pdf_path key
        if 'pdf_path' not in workflow_results and 'html_path' in workflow_results:
            html_path = workflow_results['html_path']
            # Create question_bank.pdf in same directory
            pdf_path = os.path.join(os.path.dirname(html_path), 'question_bank.pdf')
            workflow_results['pdf_path'] = pdf_path
        
        if not os.path.exists(workflow_results['pdf_path']):
            print(f"📄 PDF will be created at: {workflow_results['pdf_path']}")
            print("💡 Open the HTML file in browser and Print→Save as PDF")
        else:
            print(f"📄 Final PDF: {workflow_results['pdf_path']}")
            
        print(f"📁 LaTeX files: {workflow_results['latex_dir']}/")
        print("=" * 60)
        
    else:
        print("❌ Workflow failed")

if __name__ == "__main__":
    main()