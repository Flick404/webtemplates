# Invoice Data Extractor

Invoice data extraction system using the Donut (Document Understanding Transformer) model. This system can extract structured data from invoice PDFs and images with high accuracy.

## Features

- **AI-Powered Extraction**: Uses Donut model for accurate document understanding
- **Multiple Formats**: Supports PDF and image files (PNG, JPG, JPEG)
- **Structured Output**: Extracts data into organized JSON format
- **Web Interface**: Streamlit-based UI for easy testing and batch processing
- **Model Retraining**: Ability to retrain with new data for improved accuracy
- **Checkpoint Management**: Save and restore model states
- **Validation**: Confidence scoring and field validation
- **Export Capabilities**: Download results in JSON or raw text format

## Extracted Fields

The system extracts the following structured data from invoices:

### Header Information
- **Document Number**: Invoice or document identifier
- **Issue Date**: Date when the invoice was issued
- **Sale Date**: Date of the sale transaction
- **Order Number**: Purchase order reference (if available)
- **Buyer Information**:
  - Name
  - VAT ID
  - Address
- **Seller Information**:
  - Name
  - VAT ID
  - Address
- **Seller Bank Account**: Banking details

### Line Items
- **Item Name**: Product or service description
- **Quantity**: Number of units
- **Unit Price**: Price per unit
- **Net Value**: Net amount before tax
- **Tax Amount**: VAT or tax amount
- **Gross Value**: Total amount including tax
- **Tax Rate**: Tax percentage (e.g., "23%")

### Summary Totals
- **Total Net**: Sum of all net values
- **VAT**: Total tax amount
- **Total Gross**: Final invoice total

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- At least 8GB RAM (16GB recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd invoice_extractor_raw
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model checkpoint** (if not already present):
   - Ensure you have the `checkpoint_20250729_092137` directory in your project root
   - This contains the pre-trained Donut model for invoice extraction

## Project Structure

```
invoice_extractor_raw/
├── data/
│   ├── train/           # Training data (PDF + JSON pairs)
│   └── retrain/         # Additional training data for retraining
├── outputs/
│   └── predictions/     # Model predictions and outputs
├── src/
│   ├── parse_raw.py     # Raw output parser for Donut model
│   ├── streamlit_app.py # Web UI for testing
│   ├── retrain.py       # Model retraining script
│   ├── save_checkpoint.py # Checkpoint management
│   └── evaluate_checkpoint_regex.py # Model evaluation
├── checkpoint_*/        # Model checkpoints
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

### 1. Web Interface (Recommended)

Launch the Streamlit web application for easy testing:

```bash
cd src
streamlit run streamlit_app.py
```

**Features**:
- Upload multiple files (PDF or images)
- Real-time processing
- View both raw output and structured JSON
- Download results
- Confidence scoring and validation

### 2. Command Line Usage

#### Parse Raw Output
Test the parsing functionality:

```bash
cd src
python parse_raw.py
```

#### Evaluate Model Checkpoint
Evaluate a specific model checkpoint:

```bash
cd src
python evaluate_checkpoint_regex.py
```

#### Retrain Model
Add new training data and retrain the model:

```bash
cd src
python retrain.py
```

#### Save Checkpoint
Save the current model state:

```bash
cd src
python save_checkpoint.py
```

## Model Training and Retraining

### Training Data Format

Training data should be organized as follows:

```
data/
├── train/
│   ├── invoice1.pdf
│   ├── invoice1.json
│   ├── invoice2.pdf
│   ├── invoice2.json
│   └── ...
└── retrain/
    ├── new_invoice1.pdf
    ├── new_invoice1.json
    └── ...
```

### JSON Ground Truth Format

Each JSON file should contain the structured data:

```json
{
  "document_number": "FS/013946/01/2023",
  "issue_date": "2023-01-17",
  "sale_date": "2023-01-17",
  "buyer": {
    "name": "Comel Tomasz Hinz",
    "vat_id": "PL 8421042980",
    "address": "Szewska 11, 77-200 Miastko"
  },
  "seller": {
    "name": "IRONPACK SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ SPÓŁKA KOMANDYTOWA",
    "vat_id": "5252817909",
    "address": "Twarda 18, 00-105 Warszawa"
  },
  "invoice_line_items": [
    {
      "name": "ETYKIETY TERMICZNE BIAŁE 50x30 1000 szt",
      "quantity": 2,
      "unit_price": 5.69,
      "net_value": 9.24,
      "tax_amount": 2.14,
      "gross_value": 11.38,
      "tax_rate": "23%"
    },
    {
      "name": "KOSZT WYSYŁKI",
      "quantity": 1,
      "unit_price": 11.99,
      "net_value": 9.76,
      "tax_amount": 2.23,
      "gross_value": 11.99,
      "tax_rate": "23%"
    }
  ],
  "summary": {
    "total_net": 19.0,
    "vat": 4.37,
    "total_gross": 23.37
  },
  "seller_bank_account": "61 1140 1153 0000 3179 2000 1001",
  "order_number": null
}
```

### Retraining Process

1. **Prepare new data**: Add PDF and JSON pairs to `data/retrain/`
2. **Run retraining**: Execute `python retrain.py`
3. **Monitor progress**: The script will show training progress and final accuracy
4. **New checkpoint**: A new timestamped checkpoint will be created

## Configuration

### Model Parameters

Key parameters can be adjusted in the training scripts:

- **Learning Rate**: Default `3e-5`
- **Epochs**: Default `3-5`
- **Batch Size**: Default `2` (adjust based on GPU memory)
- **Max Length**: Default `512` tokens

### Hardware Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 10GB+ free space for models and checkpoints

## Performance and Accuracy

### Current Performance

- **Model**: Donut (Document Understanding Transformer)
- **Base Accuracy**: Varies by field (typically 70-90%)
- **Processing Speed**: ~2-5 seconds per document (GPU)
- **Supported Languages**: Primarily English and Polish invoices

### Improving Accuracy

1. **Add more training data**: Include diverse invoice formats
2. **Retrain with new data**: Use the retraining script
3. **Adjust model parameters**: Modify learning rate and epochs
4. **Preprocess images**: Ensure good image quality
5. **Validate outputs**: Use the confidence scoring system

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in training scripts
   - Use CPU instead of GPU
   - Close other GPU-intensive applications

2. **Model Loading Errors**:
   - Ensure checkpoint directory exists
   - Check file permissions
   - Verify model files are complete

3. **PDF Processing Issues**:
   - Ensure PyMuPDF is properly installed
   - Check PDF file integrity
   - Try converting PDF to image first

4. **Parsing Errors**:
   - Check raw output format
   - Verify JSON structure
   - Review regex patterns in `parse_raw.py`

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export PYTHONPATH=.
export TRANSFORMERS_VERBOSITY=info
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Donut Model**: Based on the Document Understanding Transformer from Hugging Face
- **Hugging Face**: For the transformers library and model hosting
- **Streamlit**: For the web interface framework
- **PyMuPDF**: For PDF processing capabilities

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information
4. Include error messages and system information

## Version History

- **v1.0.0**: Initial release with Donut model
- **v1.1.0**: Added Streamlit UI and retraining capabilities
- **v1.2.0**: Improved parsing and validation
- **v1.3.0**: Enhanced checkpoint management and evaluation

---

**Note**: This system is designed for invoice data extraction and may require customization for other document types or languages. 