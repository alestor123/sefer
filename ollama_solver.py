"""
Complete Ollama Question Solver with WeasyPrint PDF Generation
PRODUCTION VERSION - NO SIMULATION CODE
"""
import os
import base64
import requests
import json
from datetime import datetime

# Try WeasyPrint import
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("⚠️ WeasyPrint not available. Install with: pip install weasyprint")

class OllamaDeepSeekSolver:
    def __init__(self, images_dir='temp', latex_dir='latex_pages', 
                 ollama_url='http://localhost:11434/api/generate', 
                 model_name='llama3.1:8b'):
        self.images_dir = images_dir
        self.latex_dir = latex_dir
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Create directories
        os.makedirs(self.latex_dir, exist_ok=True)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=True)
    
    def check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model_name in model_names:
                    return True, f"✅ Connected to Ollama with {self.model_name}"
                else:
                    return False, f"❌ Model {self.model_name} not found. Available: {model_names}"
            return False, "❌ Ollama not responding"
        except Exception as e:
            return False, f"❌ Connection failed: {e}"
    
    def encode_image_base64(self, image_path):
        """Convert image to base64 for Ollama vision"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def send_image_to_ollama(self, image_path, question_number):
        """Send image to Ollama for detailed solution"""
        prompt = f"""You are the world's best mathematics and statistics tutor. Analyze this question image and provide a complete solution.

REQUIREMENTS:
1. Extract the exact question text from the image
2. Provide detailed step-by-step solution
3. Include all formulas and calculations
4. For multiple choice, identify the correct answer

FORMAT your response EXACTLY as:
Question {question_number}: [extracted question text]

Solution:
Step 1: [Identify problem type]
Step 2: [List given information]
Step 3: [Choose appropriate method]
Step 4: [Show calculations]
Step 5: [State final answer]

Answer: [Final answer with units]

Key Concepts: [List important concepts]

Use clear mathematical notation."""

        # Encode image for Ollama
        base64_img = self.encode_image_base64(image_path)
        if not base64_img:
            return None
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_img],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 4000
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                print(f"Ollama API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Failed to connect to Ollama: {e}")
            return None
    
    def process_all_images(self):
        """Process all question images using Ollama"""
        print("🚀 Starting question processing...")
        
        # Check Ollama connection
        connected, message = self.check_ollama_connection()
        print(message)
        
        if not connected:
            print("❌ Ollama not available - cannot process images")
            return []
        
        # Check images directory
        if not os.path.exists(self.images_dir):
            print(f"❌ Images directory '{self.images_dir}' not found!")
            return []
        
        image_files = sorted([f for f in os.listdir(self.images_dir) 
                            if f.lower().endswith('.png')])
        
        if not image_files:
            print("❌ No PNG images found in images directory")
            return []
        
        print(f"📸 Found {len(image_files)} question images to process")
        
        content_files = []
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(self.images_dir, img_file)
            print(f"🧠 Processing {i}/{len(image_files)}: {img_file}")
            
            content = self.send_image_to_ollama(img_path, i)
            
            if content:
                filename = f"Q{i:03d}_{img_file[:-4]}.txt"
                file_path = os.path.join(self.latex_dir, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                content_files.append(file_path)
                print(f"✅ Saved: {filename}")
            else:
                print(f"❌ Failed to process: {img_file}")
        
        return content_files
    
    def convert_to_html_pdf(self, content_files):
        """Convert content to PDF using WeasyPrint - NO PDFLATEX REQUIRED"""
        print("🔄 Converting to PDF using WeasyPrint...")
        
        if not WEASYPRINT_AVAILABLE:
            print("❌ WeasyPrint not available. Install with: pip install weasyprint")
            return None
        
        if not content_files:
            print("❌ No content files to convert")
            return None
        
        try:
            # Read all content files
            all_content = []
            for i, file_path in enumerate(content_files, 1):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix: Separate the replace operation from f-string
                content_with_breaks = content.replace('\n', '<br>')
                
                # Format content for HTML
                formatted_content = f"""
                <div class="question">
                    <h2>Question {i}</h2>
                    <div class="content">
                        {content_with_breaks}
                    </div>
                </div>
                <div class="page-break"></div>
                """
                all_content.append(formatted_content)
            
            # Fix: Separate join operation from f-string
            joined_content = ''.join(all_content)
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create HTML document
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Complete Question Bank</title>
                <style>
                    body {{
                        font-family: 'Times New Roman', serif;
                        font-size: 12pt;
                        line-height: 1.6;
                        margin: 0;
                        padding: 0;
                        color: #333;
                    }}
                    
                    .title-page {{
                        text-align: center;
                        padding: 100px 40px;
                        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                        border: 2px solid #2c3e50;
                        margin-bottom: 40px;
                    }}
                    
                    .title-page h1 {{
                        font-size: 2.5em;
                        color: #2c3e50;
                        margin-bottom: 20px;
                    }}
                    
                    .question {{
                        background: #f8f9fa;
                        border-left: 6px solid #007bff;
                        padding: 30px;
                        margin: 30px 0;
                        border-radius: 8px;
                        page-break-inside: avoid;
                    }}
                    
                    .question h2 {{
                        color: #007bff;
                        border-bottom: 2px solid #007bff;
                        padding-bottom: 10px;
                        margin-top: 0;
                    }}
                    
                    .content {{
                        background: white;
                        padding: 20px;
                        border-radius: 6px;
                        margin-top: 15px;
                    }}
                    
                    .page-break {{
                        page-break-before: always;
                    }}
                    
                    @page {{
                        margin: 1in;
                        @bottom-center {{
                            content: "Page " counter(page);
                            font-size: 10pt;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="title-page">
                    <h1>🎯 Complete Question Bank</h1>
                    <h2>Mathematics & Statistics Solutions</h2>
                    <p><strong>Generated by AI-Powered Question Solver</strong></p>
                    <p><em>Date: {current_date}</em></p>
                    <p>Total Questions: {len(content_files)}</p>
                </div>
                
                {joined_content}
                
                <div style="text-align: center; margin-top: 40px;">
                    <hr>
                    <p><strong>🎓 End of Question Bank</strong></p>
                    <p><em>Generated by AI-Powered Question Solver</em></p>
                </div>
            </body>
            </html>
            """
            
            # Generate PDF directly using WeasyPrint
            pdf_path = os.path.join(self.latex_dir, 'question_bank.pdf')
            HTML(string=html_content).write_pdf(pdf_path)
            
            if os.path.exists(pdf_path):
                print(f"✅ PDF successfully generated: {pdf_path}")
                return pdf_path
            else:
                print("❌ PDF file was not created")
                return None
                
        except Exception as e:
            print(f"❌ WeasyPrint conversion failed: {e}")
            return None
    
    def run_complete_workflow(self):
        """Execute the complete workflow"""
        print("🎯 STARTING COMPLETE WORKFLOW")
        print("=" * 60)
        
        # Step 1: Process images with Ollama
        content_files = self.process_all_images()
        
        if not content_files:
            print("❌ No content generated from images")
            return None
        
        # Step 2: Convert to PDF
        pdf_path = self.convert_to_html_pdf(content_files)
        
        if pdf_path:
            print("=" * 60)
            print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"📄 Final PDF: {os.path.basename(pdf_path)}")
            print(f"📖 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
            print(f"📁 Content files: {self.latex_dir}/")
            print(f"🎓 Open PDF: {pdf_path}")
            print("=" * 60)
            
            return {
                "pdf_path": pdf_path,
                "latex_dir": self.latex_dir,
                "total_questions": len(content_files),
                "format": "PDF"
            }
        else:
            print("❌ PDF generation failed")
            return None

# Usage example:
if __name__ == "__main__":
    solver = OllamaDeepSeekSolver()
    results = solver.run_complete_workflow()
    
    if results:
        print(f"\n🎓 SUCCESS! Generated question bank PDF with {results['total_questions']} questions")
        print(f"📖 Open this file: {results['pdf_path']}")
    else:
        print("\n❌ Workflow failed")
