"""
Question Extraction Module
Structure-aware PDF question extractor for exam papers
"""

import os
import fitz  # PyMuPDF
import requests
import json
import re
import datetime

class QuestionExtractor:
    def __init__(self, model_name="llama3.2:1b", output_dir="temp"):
        """Initialize the question extractor"""
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.output_dir = output_dir
        
        # Structural patterns for exam format
        self.question_delimiter_pattern = r'Question Number\s*:\s*(\d+)\s+Question Id\s*:\s*(\d+)'
        self.question_block_pattern = r'(Question Number\s*:.*?)(?=Question Number\s*:|$)'
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_ollama_status(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                return {"status": False, "message": "Ollama is not running"}
                
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                return {
                    "status": False, 
                    "message": f"Model {self.model_name} not found. Available: {', '.join(model_names)}"
                }
                
            return {"status": True, "message": "Ollama ready"}
        except Exception as e:
            return {"status": False, "message": f"Connection error: {str(e)}"}
    
    def query_ollama(self, prompt, timeout=30):
        """Send query to Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9}
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=timeout)
            if response.status_code == 200:
                return json.loads(response.text)["response"]
        except Exception as e:
            print(f"Ollama query failed: {e}")
        
        return None
    
    def extract_question_blocks(self, text):
        """Extract complete question blocks using structural delimiters"""
        question_blocks = re.findall(self.question_block_pattern, text, re.DOTALL | re.IGNORECASE)
        
        extracted_questions = []
        
        for i, block in enumerate(question_blocks):
            delimiter_match = re.search(self.question_delimiter_pattern, block, re.IGNORECASE)
            
            if delimiter_match:
                question_number = delimiter_match.group(1)
                question_id = delimiter_match.group(2)
                
                # Extract the actual question text
                question_text = self.extract_question_text(block)
                
                extracted_questions.append({
                    'question_number': int(question_number),
                    'question_id': question_id,
                    'question_text': question_text,
                    'full_block': block,
                    'block_index': i
                })
        
        return extracted_questions
    
    def extract_question_text(self, block):
        """Extract question text from block using patterns + AI"""
        # Try pattern-based extraction first
        patterns = [
            r'Question Label\s*:\s*[^\n]*\n\n([^\n]+\?)',
            r'Based on.*?answer.*?\n\n([^\n]+\?)',
            r'([A-Z][^\n]*\?)',
            r'What is.*?\?',
            r'How (?:much|many).*?\?',
            r'Which of.*?\?',
            r'In how many.*?\?',
            r'If .*?\?',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, block, re.IGNORECASE | re.DOTALL)
            if matches:
                question = max(matches, key=len).strip()
                if len(question) > 15 and '?' in question:
                    return question
        
        # Fallback to AI extraction
        return self.extract_question_with_ai(block)
    
    def extract_question_with_ai(self, block):
        """Use AI to extract question from complex blocks"""
        prompt = f"""Extract the main question from this exam block.
Return only the question text, nothing else.

Block:
{block[:1000]}

Question:"""
        
        response = self.query_ollama(prompt)
        if response:
            question = response.strip()
            question = re.sub(r'^(Question:|Answer:|The question is:?)', '', question, flags=re.IGNORECASE).strip()
            
            if len(question) > 10 and not question.lower().startswith('i cannot'):
                return question
        
        return f"Question {block[:50]}..."  # Fallback
    
    def find_question_on_page(self, page, question_number, question_id):
        """Find question location on page"""
        search_strategies = [
            f"Question Number : {question_number}",
            f"Question Id : {question_id}",
        ]
        
        for strategy in search_strategies:
            instances = page.search_for(strategy)
            if instances:
                return instances[0]
        
        return None
    
    def extract_question_image(self, page, question_bbox, next_question_bbox, output_path):
        """Extract high-quality question image"""
        try:
            page_rect = page.rect
            
            if not question_bbox:
                # Full page fallback
                clip_rect = fitz.Rect(0, 0, page_rect.width, page_rect.height)
            else:
                # Smart cropping
                x0 = 0
                y0 = max(0, question_bbox.y0 - 20)
                x1 = page_rect.width
                
                if next_question_bbox:
                    y1 = max(question_bbox.y1 + 50, next_question_bbox.y0 - 15)
                else:
                    y1 = min(page_rect.height, question_bbox.y1 + 300)
                
                clip_rect = fitz.Rect(x0, y0, x1, y1)
            
            # High-quality rendering
            mat = fitz.Matrix(2.5, 2.5)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            
            # Save image
            pix.save(output_path)
            pix = None
            
            return True
            
        except Exception as e:
            print(f"Image extraction failed: {e}")
            return False
    
    def process_pdf(self, pdf_path):
        """Main processing function"""
        results = {
            "success": False,
            "message": "",
            "questions_found": 0,
            "questions_extracted": 0,
            "output_files": [],
            "report_path": "",
            "viewer_path": ""
        }
        
        # Check Ollama
        ollama_status = self.check_ollama_status()
        if not ollama_status["status"]:
            results["message"] = f"Ollama Error: {ollama_status['message']}"
            return results
        
        # Open PDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            results["message"] = f"Cannot open PDF: {e}"
            return results
        
        # Extract full document text
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        
        # Extract question blocks
        question_blocks = self.extract_question_blocks(full_text)
        results["questions_found"] = len(question_blocks)
        
        if not question_blocks:
            results["message"] = "No questions found using structural pattern"
            doc.close()
            return results
        
        # Process each question
        extracted_questions = []
        
        for i, question_data in enumerate(question_blocks):
            # Find page containing this question
            question_page = None
            question_page_num = None
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                search_text = f"Question Number : {question_data['question_number']}"
                if page.search_for(search_text):
                    question_page = page
                    question_page_num = page_num
                    break
            
            if not question_page:
                continue
            
            # Get question and next question locations
            question_bbox = self.find_question_on_page(
                question_page, 
                question_data['question_number'], 
                question_data['question_id']
            )
            
            next_question_bbox = None
            if i + 1 < len(question_blocks):
                next_data = question_blocks[i + 1]
                # Try to find next question on same page
                next_question_bbox = self.find_question_on_page(
                    question_page,
                    next_data['question_number'],
                    next_data['question_id']
                )
            
            # Create filename
            safe_text = re.sub(r'[^\w\s-]', '', question_data['question_text'][:40])
            safe_text = re.sub(r'[-\s]+', '_', safe_text).strip('_')
            
            img_filename = f"Q{question_data['question_number']:03d}_P{question_page_num + 1}_{safe_text}.png"
            img_path = os.path.join(self.output_dir, img_filename)
            
            # Extract image
            success = self.extract_question_image(
                question_page, question_bbox, next_question_bbox, img_path
            )
            
            if success:
                extracted_questions.append({
                    'question_number': question_data['question_number'],
                    'question_id': question_data['question_id'],
                    'page': question_page_num + 1,
                    'question_text': question_data['question_text'],
                    'filename': img_filename,
                    'file_path': img_path
                })
                results["output_files"].append(img_path)
        
        doc.close()
        
        results["questions_extracted"] = len(extracted_questions)
        
        # Generate reports
        report_info = self.generate_reports(extracted_questions, pdf_path)
        results["report_path"] = report_info["report_path"]
        results["viewer_path"] = report_info["viewer_path"]
        
        results["success"] = True
        results["message"] = f"Successfully extracted {len(extracted_questions)} questions"
        
        return results
    
    def generate_reports(self, questions, pdf_path):
        """Generate text report and HTML viewer"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Text report
        report_path = os.path.join(self.output_dir, "extraction_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PDF QUESTION EXTRACTION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"PDF File: {pdf_path}\n")
            f.write(f"Extraction Time: {datetime.datetime.now()}\n")
            f.write(f"Total Questions: {len(questions)}\n")
            f.write("=" * 50 + "\n\n")
            
            for q in questions:
                f.write(f"Q{q['question_number']:03d} | Page {q['page']}\n")
                f.write(f"ID: {q['question_id']}\n")
                f.write(f"File: {q['filename']}\n")
                f.write(f"Text: {q['question_text']}\n")
                f.write("-" * 30 + "\n\n")
        
        # HTML viewer
        viewer_path = os.path.join(self.output_dir, "viewer.html")
        with open(viewer_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html><head>
<title>Question Extraction Results</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
.container { max-width: 1000px; margin: 0 auto; }
.header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
.question { background: white; margin: 15px 0; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.question img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
h1 { color: #333; }
</style>
</head><body>
<div class="container">
<div class="header">
<h1>📚 Question Extraction Results</h1>
<p><strong>Total Questions:</strong> """ + str(len(questions)) + """</p>
<p><strong>Extraction Time:</strong> """ + str(datetime.datetime.now()) + """</p>
</div>
""")
            
            for q in questions:
                f.write(f"""
<div class="question">
<h3>Question #{q['question_number']:03d} - Page {q['page']}</h3>
<p><strong>ID:</strong> {q['question_id']}</p>
<p><strong>Question:</strong> {q['question_text']}</p>
<img src="{q['filename']}" alt="Question {q['question_number']}">
</div>
""")
            
            f.write("</div></body></html>")
        
        return {
            "report_path": report_path,
            "viewer_path": viewer_path
        }
    
    def clean_temp_directory(self):
        """Clean the temp directory"""
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
