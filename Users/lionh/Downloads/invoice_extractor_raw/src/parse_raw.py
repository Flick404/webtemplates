import re
import json
from typing import Dict, Optional

class RawOutputParser:
    """Parse Donut model's raw output into structured JSON"""
    
    def __init__(self):
        # Define field patterns with proper boundaries
        self.patterns = {
            "document_number": r"Document Number:\s*([^I]+?)(?=\s*Issue Date:)",
            "issue_date": r"Issue Date:\s*([^S]+?)(?=\s*Sale Date:)",
            "sale_date": r"Sale Date:\s*([^B]+?)(?=\s*Buyer Name:)",
            "buyer_name": r"Buyer Name:\s*([^B]+?)(?=\s*Buyer VAT ID:)",
            "buyer_vat_id": r"Buyer VAT ID:\s*([^B]+?)(?=\s*Buyer Address:)",
            "buyer_address": r"Buyer Address:\s*([^S]+?)(?=\s*Seller Name:)",
            "seller_name": r"Seller Name:\s*([^S]+?)(?=\s*Seller VAT ID:)",
            "seller_vat_id": r"Seller VAT ID:\s*([^S]+?)(?=\s*Seller Address:)",
            "seller_address": r"Seller Address:\s*([^S]+?)(?=\s*Seller Bank Account:)",
            "seller_bank_account": r"Seller Bank Account:\s*([^\s]*)",
            "order_number": r"Order Number:\s*([^\s]*)",
            "total_net": r"Total Net:\s*([^\s]*)",
            "vat": r"VAT:\s*([^\s]*)",
            "total_gross": r"Total Gross:\s*([^\s]*)"
        }
        
        # Alternative patterns for when some fields are missing
        self.fallback_patterns = {
            "document_number": r"Document Number:\s*([^I]+?)(?=\s*(?:Issue Date:|$))",
            "issue_date": r"Issue Date:\s*([^S]+?)(?=\s*(?:Sale Date:|$))",
            "sale_date": r"Sale Date:\s*([^B]+?)(?=\s*(?:Buyer Name:|$))",
            "buyer_name": r"Buyer Name:\s*([^B]+?)(?=\s*(?:Buyer VAT ID:|$))",
            "buyer_vat_id": r"Buyer VAT ID:\s*([^B]+?)(?=\s*(?:Buyer Address:|$))",
            "buyer_address": r"Buyer Address:\s*([^S]+?)(?=\s*(?:Seller Name:|$))",
            "seller_name": r"Seller Name:\s*([^S]+?)(?=\s*(?:Seller VAT ID:|$))",
            "seller_vat_id": r"Seller VAT ID:\s*([^S]+?)(?=\s*(?:Seller Address:|$))",
            "seller_address": r"Seller Address:\s*([^S]+?)(?=\s*(?:Seller Bank Account:|$))",
            "seller_bank_account": r"Seller Bank Account:\s*([^\s]*)",
            "order_number": r"Order Number:\s*([^\s]*)",
            "total_net": r"Total Net:\s*([^\s]*)",
            "vat": r"VAT:\s*([^\s]*)",
            "total_gross": r"Total Gross:\s*([^\s]*)"
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize the raw text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove any special tokens or artifacts
        text = re.sub(r'<[^>]+>', '', text)  # Remove XML-like tags
        text = re.sub(r'\[[^\]]+\]', '', text)  # Remove bracket content
        
        return text
    
    def extract_field(self, text: str, field_name: str) -> str:
        """Extract a single field using both primary and fallback patterns"""
        text = self.clean_text(text)
        
        # Try primary pattern first
        pattern = self.patterns.get(field_name)
        if pattern:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result = match.group(1)
                return result.strip() if isinstance(result, str) else str(result)
        
        # Try fallback pattern
        fallback_pattern = self.fallback_patterns.get(field_name)
        if fallback_pattern:
            match = re.search(fallback_pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result = match.group(1)
                return result.strip() if isinstance(result, str) else str(result)
        
        return ""
    
    def extract_line_items(self, raw_text: str) -> list:
        """Extract line items from raw text"""
        line_items = []
        
        # Look for line item patterns
        # Pattern: Item Name: [name] Quantity: [qty] Unit Price: [price] Net Value: [net] Tax: [tax] Gross: [gross] Tax Rate: [rate]
        line_item_pattern = r"Item Name:\s*([^Q]+?)\s*Quantity:\s*([^U]+?)\s*Unit Price:\s*([^N]+?)\s*Net Value:\s*([^T]+?)\s*Tax:\s*([^G]+?)\s*Gross:\s*([^T]+?)\s*Tax Rate:\s*([^\s]+)"
        
        matches = re.finditer(line_item_pattern, raw_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            line_item = {
                "name": match.group(1).strip(),
                "quantity": self.parse_number(match.group(2)),
                "unit_price": self.parse_number(match.group(3)),
                "net_value": self.parse_number(match.group(4)),
                "tax_amount": self.parse_number(match.group(5)),
                "gross_value": self.parse_number(match.group(6)),
                "tax_rate": match.group(7).strip()
            }
            line_items.append(line_item)
        
        return line_items
    
    def parse_number(self, text: str) -> float:
        """Parse number from text, handling various formats"""
        if not text:
            return 0.0
        
        # Remove currency symbols and extra spaces
        cleaned = re.sub(r'[^\d.,]', '', text.strip())
        
        # Handle different decimal separators
        if ',' in cleaned and '.' in cleaned:
            # Format like "1,234.56" or "1.234,56"
            if cleaned.rfind('.') > cleaned.rfind(','):
                # "1,234.56" format
                cleaned = cleaned.replace(',', '')
            else:
                # "1.234,56" format
                cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            # Assume comma is decimal separator
            cleaned = cleaned.replace(',', '.')
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def parse_raw_output(self, raw_text: str) -> Dict:
        """Parse raw Donut output into structured JSON"""
        result = {
            "document_number": "",
            "issue_date": "",
            "sale_date": "",
            "buyer": {
                "name": "",
                "vat_id": "",
                "address": ""
            },
            "seller": {
                "name": "",
                "vat_id": "",
                "address": ""
            },
            "invoice_line_items": [],
            "summary": {
                "total_net": "",
                "vat": "",
                "total_gross": ""
            },
            "seller_bank_account": "",
            "order_number": ""
        }
        
        # Extract header fields
        result["document_number"] = self.extract_field(raw_text, "document_number")
        result["issue_date"] = self.extract_field(raw_text, "issue_date")
        result["sale_date"] = self.extract_field(raw_text, "sale_date")
        result["buyer"]["name"] = self.extract_field(raw_text, "buyer_name")
        result["buyer"]["vat_id"] = self.extract_field(raw_text, "buyer_vat_id")
        result["buyer"]["address"] = self.extract_field(raw_text, "buyer_address")
        result["seller"]["name"] = self.extract_field(raw_text, "seller_name")
        result["seller"]["vat_id"] = self.extract_field(raw_text, "seller_vat_id")
        result["seller"]["address"] = self.extract_field(raw_text, "seller_address")
        result["seller_bank_account"] = self.extract_field(raw_text, "seller_bank_account")
        result["order_number"] = self.extract_field(raw_text, "order_number")
        
        # Extract line items
        result["invoice_line_items"] = self.extract_line_items(raw_text)
        
        # Extract summary totals
        result["summary"]["total_net"] = self.extract_field(raw_text, "total_net")
        result["summary"]["vat"] = self.extract_field(raw_text, "vat")
        result["summary"]["total_gross"] = self.extract_field(raw_text, "total_gross")
        
        return result
    
    def validate_parsing(self, raw_text: str, parsed_json: Dict) -> Dict:
        """Validate the parsing and provide confidence scores"""
        validation = {
            "raw_text_length": len(raw_text),
            "fields_extracted": 0,
            "confidence_score": 0.0,
            "missing_fields": [],
            "warnings": []
        }
        
        # Count extracted fields
        total_fields = 0
        extracted_fields = 0
        
        for field_name, value in parsed_json.items():
            if field_name in ["buyer", "seller"]:
                for subfield, subvalue in value.items():
                    total_fields += 1
                    if subvalue.strip():
                        extracted_fields += 1
                    else:
                        validation["missing_fields"].append(f"{field_name}.{subfield}")
            else:
                total_fields += 1
                if value.strip():
                    extracted_fields += 1
                else:
                    validation["missing_fields"].append(field_name)
        
        validation["fields_extracted"] = extracted_fields
        validation["confidence_score"] = extracted_fields / total_fields if total_fields > 0 else 0.0
        
        # Add warnings for potential issues
        if validation["confidence_score"] < 0.5:
            validation["warnings"].append("Low confidence - many fields missing")
        
        if len(raw_text) < 50:
            validation["warnings"].append("Very short raw output - possible extraction failure")
        
        return validation

def main():
    """Test the parser with sample outputs"""
    parser = RawOutputParser()
    
    # Test cases
    test_cases = [
        "Document Number: FS 7/Hostel/2016 Issue Date: 2016-01-13 Sale Date: 2016-01-12 Buyer Name: Comel - Tomasz Hinz Buyer VAT ID: 842-104-29-80 Buyer Address: ul. Szewska 11, 77-200 Miastko Seller Name: Rajłwa y Hostel sp. z o.o. Seller VAT ID: 7543057261 Seller Address: Krzysztof Seller Bank Account: 04 1950 0001 2006 6809 0107 0002",
        "Document Number: 948/A/01/2016 Issue Date: 2016-01-18 Sale Date: 2016-01-18 Buyer Name: Comel Buyer VAT ID: 8421042980 Buyer Address: Szewska 11, 77-200 Miastko Seller Name: Nokaut.pl Sp. z o.o. Seller VAT ID: 701-036-03-01 Seller Address: ul. Jodłowa 1/3, 81-526 Gdynia Seller Bank Account: ul. Jodłowa 173, 81-526 Gdynia"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Raw: {test_case}")
        
        parsed = parser.parse_raw_output(test_case)
        validation = parser.validate_parsing(test_case, parsed)
        
        print(f"Parsed: {json.dumps(parsed, indent=2, ensure_ascii=False)}")
        print(f"Validation: {json.dumps(validation, indent=2)}")

if __name__ == "__main__":
    main() 