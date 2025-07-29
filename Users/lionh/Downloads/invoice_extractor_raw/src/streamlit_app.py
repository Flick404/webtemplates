import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import fitz
import io
import json
import tempfile
import os
from transformers import DonutProcessor, VisionEncoderDecoderModel
from parse_raw import RawOutputParser

class InvoiceExtractor:
    def __init__(self, model_path: str = "checkpoint_20250729_092137"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.processor = DonutProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize parser
        self.parser = RawOutputParser()
        
    def load_image(self, file) -> Image.Image:
        """Load image from uploaded file"""
        if file.type == "application/pdf":
            # Handle PDF
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[0]
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("ppm"))).convert("RGB")
            doc.close()
        else:
            # Handle image
            img = Image.open(file).convert("RGB")
        
        return img
    
    def extract_invoice_data(self, image: Image.Image) -> tuple:
        """Extract invoice data from image"""
        # Prepare input
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=512,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        
        # Decode generated text
        raw_text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
        
        # Parse into JSON using the RawOutputParser
        parsed_json = self.parser.parse_raw_output(raw_text)
        validation = self.parser.validate_parsing(raw_text, parsed_json)
        
        return raw_text, parsed_json, validation

def main():
    st.set_page_config(
        page_title="Invoice Extractor",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Invoice Data Extractor")
    st.markdown("Extract structured data from invoice images and PDFs using AI")
    
    # Initialize extractor
    @st.cache_resource
    def load_extractor():
        try:
            return InvoiceExtractor()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    
    extractor = load_extractor()
    
    if extractor is None:
        st.error("Model not loaded. Please check if the checkpoint exists.")
        return
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # File upload
    st.sidebar.subheader("Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose invoice files",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload PDF or image files of invoices"
    )
    
    # Model info
    st.sidebar.subheader("Model Information")
    st.sidebar.info(f"Device: {extractor.device}")
    st.sidebar.info("Model: Donut (Document Understanding)")
    
    # Main content
    if uploaded_files:
        st.header("Processing Results")
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"File {i+1}: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display image
                try:
                    image = extractor.load_image(uploaded_file)
                    st.image(image, caption="Uploaded Document", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    continue
            
            with col2:
                # Process and display results
                with st.spinner("Extracting data..."):
                    try:
                        raw_text, parsed_json, validation = extractor.extract_invoice_data(image)
                        
                        # Display results in tabs
                        tab1, tab2, tab3 = st.tabs(["Structured Data", "Raw Output", "Validation"])
                        
                        with tab1:
                            st.json(parsed_json)
                            
                            # Download JSON
                            json_str = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"{uploaded_file.name}_extracted.json",
                                mime="application/json"
                            )
                        
                        with tab2:
                            st.text_area("Raw Model Output", raw_text, height=200)
                            
                            # Download raw text
                            st.download_button(
                                label="Download Raw Text",
                                data=raw_text,
                                file_name=f"{uploaded_file.name}_raw.txt",
                                mime="text/plain"
                            )
                        
                        with tab3:
                            # Confidence score
                            confidence = validation["confidence_score"]
                            st.metric("Confidence Score", f"{confidence:.2%}")
                            
                            # Fields extracted
                            st.metric("Fields Extracted", f"{validation['fields_extracted']}")
                            
                            # Missing fields
                            if validation["missing_fields"]:
                                st.warning("Missing Fields:")
                                for field in validation["missing_fields"]:
                                    st.write(f"• {field}")
                            
                            # Warnings
                            if validation["warnings"]:
                                st.error("Warnings:")
                                for warning in validation["warnings"]:
                                    st.write(f"• {warning}")
                            
                            # Validation details
                            with st.expander("Detailed Validation"):
                                st.json(validation)
                        
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            
            st.divider()
    
    else:
        # Welcome message
        st.header("Welcome to Invoice Extractor!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How to use:
            1. **Upload Files**: Use the sidebar to upload invoice PDFs or images
            2. **Process**: The AI will automatically extract structured data
            3. **Review**: View both raw output and parsed JSON
            4. **Download**: Save results in your preferred format
            
            ### Supported Formats:
            - PDF files
            - PNG images
            - JPG/JPEG images
            """)
        
        with col2:
            st.markdown("""
            ### Extracted Fields:
            - **Document Number**
            - **Issue Date**
            - **Sale Date**
            - **Buyer Information** (Name, VAT ID, Address)
            - **Seller Information** (Name, VAT ID, Address)
            - **Seller Bank Account**
            
            ### Features:
            - Real-time processing
            - Confidence scoring
            - Field validation
            - Export capabilities
            """)
        
        # Example output
        st.subheader("Example Output")
        example_json = {
            "document_number": "FS 7/Hostel/2016",
            "issue_date": "2016-01-13",
            "sale_date": "2016-01-12",
            "buyer": {
                "name": "Comel - Tomasz Hinz",
                "vat_id": "842-104-29-80",
                "address": "ul. Szewska 11, 77-200 Miastko"
            },
            "seller": {
                "name": "Rajłwa y Hostel sp. z o.o.",
                "vat_id": "7543057261",
                "address": "Krzysztof"
            },
            "seller_bank_account": "04 1950 0001 2006 6809 0107 0002"
        }
        
        st.json(example_json)

if __name__ == "__main__":
    main() 