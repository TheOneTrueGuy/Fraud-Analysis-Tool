import os
import logging
import pandas as pd
import nltk
import kagglehub
import google.generativeai as genai
from google.generativeai import caching
from typing import Dict, List, Optional
import datetime
from setup_nltk import setup_nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_caches_from_csv(file_path: str) -> List[caching.CachedContent]:
    """Create context caches from CSV data."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("CSV file is empty")
        
        required_columns = ['sender', 'recipient', 'subject', 'body']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create caches for batches of emails
        caches = []
        batch_size = 100  # Adjust based on your needs
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            cache = caching.CachedContent.create(
                model="gemini-1.5-flash",
                display_name=f'email_batch_{i//batch_size}',
                content=batch.to_dict('records'),
                ttl=datetime.timedelta(hours=1)
            )
            caches.append(cache)
        return caches
    
    except pd.errors.EmptyDataError:
        logging.error("CSV file is empty")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing CSV: {str(e)}")
        raise

def analyze_file(file, case_details: str, keywords: Optional[str] = None) -> pd.DataFrame:
    """Analyze a file for potential fraud evidence."""
    try:
        # Ensure NLTK resources are available
        setup_nltk()
        
        # Handle file input
        if file is None:
            path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
            file_path = os.path.join(path, 'emails.csv')
            logging.info(f"Using default email dataset from: {file_path}")
        else:
            file_path = file.name
            logging.info(f"Using uploaded file: {file_path}")
        
        # Create caches and process emails
        caches = create_caches_from_csv(file_path)
        df = pd.read_csv(file_path)
        sample_emails = df.head().to_dict('records')
        results = process_email_batch(sample_emails, caches)
        
        # Convert results to DataFrame
        output_data = []
        for result in results:
            row = {
                'File': result.get('file', ''),
                'Status': result.get('status', ''),
                'Analysis': result.get('analysis', result.get('error', ''))
            }
            output_data.append(row)
        
        results_df = pd.DataFrame(output_data)
        
        # Highlight keywords if provided
        if keywords:
            for keyword in keywords.split(','):
                keyword = keyword.strip()
                if keyword:
                    results_df['Analysis'] = results_df['Analysis'].str.replace(
                        f'({keyword})',
                        r'<mark>\\1</mark>',
                        case=False,
                        regex=True
                    )
        
        return results_df
    
    except Exception as e:
        logging.error(f"Error in analyze_file: {str(e)}")
        raise

def process_email_batch(emails: List[Dict], caches: List[caching.CachedContent]) -> List[Dict]:
    """Process a batch of emails using the cached model."""
    results = []
    for email in emails:
        for attempt in range(3):  # Retry logic
            try:
                model = genai.GenerativeModel.from_cached_content(caches[0])
                response = model.generate_content([
                    f"Analyze this email for potential fraud evidence:\n\n"
                    f"From: {email.get('sender', 'Unknown')}\n"
                    f"To: {email.get('recipient', 'Unknown')}\n"
                    f"Subject: {email.get('subject', 'No Subject')}\n"
                    f"Body: {email.get('body', 'No Content')}"
                ])
                
                results.append({
                    'file': f"{email.get('sender', 'Unknown')} - {email.get('subject', 'No Subject')}",
                    'status': 'success',
                    'analysis': response.text
                })
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logging.error(f"Failed to process email after 3 attempts: {str(e)}")
                    results.append({
                        'file': f"{email.get('sender', 'Unknown')} - {email.get('subject', 'No Subject')}",
                        'status': 'failed',
                        'error': str(e)
                    })
    
    return results

# Create the Gradio interface
def create_interface():
    import gradio as gr
    
    with gr.Blocks() as demo:
        gr.Markdown("# Fraud Analysis Tool")
        gr.Markdown("*Upload a CSV file or leave empty to use the default dataset*")
        
        with gr.Row():
            file_input = gr.File(label="Upload CSV File (Optional)")
            case_details = gr.Textbox(label="Case Details", placeholder="Enter case details...")
        
        with gr.Row():
            analyze_btn = gr.Button("Analyze")
            keyword_input = gr.Textbox(
                label="Highlight Keywords",
                placeholder="Enter keywords to highlight (comma-separated)..."
            )
        
        output_table = gr.DataFrame(
            headers=["File", "Status", "Analysis"],
            datatype=["str", "str", "markdown"],
            label="Analysis Results"
        )
        
        analyze_btn.click(
            analyze_file,
            inputs=[file_input, case_details, keyword_input],
            outputs=output_table
        )
    
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(debug=True)  # Enable debug mode for better error tracking
