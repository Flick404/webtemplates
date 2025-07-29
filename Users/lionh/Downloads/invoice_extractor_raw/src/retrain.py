import json
import os
import torch
import shutil
from pathlib import Path
from PIL import Image
import fitz
import io
from transformers import DonutProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer
from datasets import Dataset
import re
import datetime
from parse_raw import RawOutputParser

class InvoiceRetrainer:
    def __init__(self, base_checkpoint: str = "checkpoint_20250729_092137"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load base model and processor
        self.processor = DonutProcessor.from_pretrained(base_checkpoint)
        self.model = VisionEncoderDecoderModel.from_pretrained(base_checkpoint)
        self.model.to(self.device)
        
        # Set decoder start token
        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.max_length = 512
        
        self.parser = RawOutputParser()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.lower().strip())
        
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path (PDF or image file)"""
        path = Path(image_path)
        
        if path.suffix.lower() == ".pdf":
            doc = fitz.open(str(path))
            page = doc[0]
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("ppm"))).convert("RGB")
            doc.close()
        else:
            img = Image.open(path).convert("RGB")
        
        return img
    
    def format_ground_truth(self, gt: dict) -> str:
        """Format ground truth into a structured text format"""
        lines = []
        
        # Document info
        if gt.get("document_number"):
            lines.append(f"Document Number: {gt['document_number']}")
        if gt.get("issue_date"):
            lines.append(f"Issue Date: {gt['issue_date']}")
        if gt.get("sale_date"):
            lines.append(f"Sale Date: {gt['sale_date']}")
        
        # Buyer info
        buyer = gt.get("buyer", {})
        if buyer.get("name"):
            lines.append(f"Buyer Name: {buyer['name']}")
        if buyer.get("vat_id"):
            lines.append(f"Buyer VAT ID: {buyer['vat_id']}")
        if buyer.get("address"):
            lines.append(f"Buyer Address: {buyer['address']}")
        
        # Seller info
        seller = gt.get("seller", {})
        if seller.get("name"):
            lines.append(f"Seller Name: {seller['name']}")
        if seller.get("vat_id"):
            lines.append(f"Seller VAT ID: {seller['vat_id']}")
        if seller.get("address"):
            lines.append(f"Seller Address: {seller['address']}")
        
        # Bank account
        if gt.get("seller_bank_account"):
            lines.append(f"Seller Bank Account: {gt['seller_bank_account']}")
        
        # Order number
        if gt.get("order_number"):
            lines.append(f"Order Number: {gt['order_number']}")
        
        # Line items
        line_items = gt.get("invoice_line_items", [])
        for i, item in enumerate(line_items, 1):
            lines.append(f"Item Name: {item.get('name', '')}")
            lines.append(f"Quantity: {item.get('quantity', '')}")
            lines.append(f"Unit Price: {item.get('unit_price', '')}")
            lines.append(f"Net Value: {item.get('net_value', '')}")
            lines.append(f"Tax: {item.get('tax_amount', '')}")
            lines.append(f"Gross: {item.get('gross_value', '')}")
            lines.append(f"Tax Rate: {item.get('tax_rate', '')}")
            if i < len(line_items):
                lines.append("---")  # Separator between items
        
        # Summary totals
        summary = gt.get("summary", {})
        if summary.get("total_net"):
            lines.append(f"Total Net: {summary['total_net']}")
        if summary.get("vat"):
            lines.append(f"VAT: {summary['vat']}")
        if summary.get("total_gross"):
            lines.append(f"Total Gross: {summary['total_gross']}")
        
        return "\n".join(lines)
    
    def prepare_retrain_dataset(self, original_data_dir: str = "data/train", retrain_data_dir: str = "data/retrain") -> Dataset:
        """Prepare dataset combining original and new retrain data"""
        print("🔄 Preparing retrain dataset...")
        
        dataset_data = []
        
        # Process original data
        original_dir = Path(original_data_dir)
        if original_dir.exists():
            print(f"📁 Processing original data from {original_data_dir}")
            original_files = list(original_dir.glob("*.pdf"))
            for pdf_file in original_files:
                json_file = pdf_file.with_suffix(".json")
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            gt = json.load(f)
                        
                        image = self.load_image(str(pdf_file))
                        target_text = self.format_ground_truth(gt)
                        
                        if target_text.strip():
                            dataset_data.append({
                                "image": image,
                                "text": target_text,
                                "source": "original"
                            })
                    except Exception as e:
                        print(f"⚠️ Error processing {pdf_file.name}: {e}")
        
        # Process retrain data
        retrain_dir = Path(retrain_data_dir)
        if retrain_dir.exists():
            print(f"📁 Processing retrain data from {retrain_data_dir}")
            retrain_files = list(retrain_dir.glob("*.pdf"))
            for pdf_file in retrain_files:
                json_file = pdf_file.with_suffix(".json")
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            gt = json.load(f)
                        
                        image = self.load_image(str(pdf_file))
                        target_text = self.format_ground_truth(gt)
                        
                        if target_text.strip():
                            dataset_data.append({
                                "image": image,
                                "text": target_text,
                                "source": "retrain"
                            })
                    except Exception as e:
                        print(f"⚠️ Error processing {pdf_file.name}: {e}")
        else:
            print(f"⚠️ Retrain directory {retrain_data_dir} not found")
        
        print(f"✅ Prepared {len(dataset_data)} samples for retraining")
        print(f"  - Original: {len([d for d in dataset_data if d['source'] == 'original'])}")
        print(f"  - Retrain: {len([d for d in dataset_data if d['source'] == 'retrain'])}")
        
        # Create dataset
        dataset = Dataset.from_list(dataset_data)
        return dataset
    
    def preprocess_function(self, examples):
        """Preprocess function for Donut training"""
        images = examples["image"]
        texts = examples["text"]
        
        # Process images and texts
        pixel_values = self.processor(images, return_tensors="pt").pixel_values
        target_encodings = self.processor.tokenizer(
            texts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = target_encodings.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    
    def create_checkpoint_name(self) -> str:
        """Create a timestamped checkpoint name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_retrain_{timestamp}"
    
    def retrain_model(self, epochs: int = 5, learning_rate: float = 3e-5) -> str:
        """Retrain the model with new data"""
        print("🚀 Starting retraining process...")
        
        # Prepare dataset
        dataset = self.prepare_retrain_dataset()
        
        if len(dataset) == 0:
            print("❌ No data found for retraining!")
            return None
        
        # Preprocess dataset
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        train_dataset = processed_dataset.select(range(int(0.8 * len(processed_dataset))))
        eval_dataset = processed_dataset.select(range(int(0.8 * len(processed_dataset)), len(processed_dataset)))
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        
        # Create checkpoint name
        checkpoint_name = self.create_checkpoint_name()
        checkpoint_path = f"./{checkpoint_name}"
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=checkpoint_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=20,
            save_steps=50,
            logging_steps=10,
            learning_rate=learning_rate,
            warmup_steps=50,
            weight_decay=0.01,
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            push_to_hub=False,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        print(f"🎯 Training for {epochs} epochs...")
        trainer.train()
        
        # Save final model
        trainer.save_model(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        
        # Save training info
        training_info = {
            "checkpoint_name": checkpoint_name,
            "training_date": datetime.datetime.now().isoformat(),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "total_samples": len(dataset),
            "original_samples": len([d for d in dataset if d['source'] == 'original']),
            "retrain_samples": len([d for d in dataset if d['source'] == 'retrain']),
            "device": str(self.device)
        }
        
        with open(f"{checkpoint_path}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"✅ Retraining completed! Checkpoint saved to: {checkpoint_name}")
        return checkpoint_name
    
    def evaluate_retrained_model(self, checkpoint_path: str, test_files: int = 10) -> dict:
        """Evaluate the retrained model"""
        print(f"🔍 Evaluating retrained model: {checkpoint_path}")
        
        # Load retrained model
        processor = DonutProcessor.from_pretrained(checkpoint_path)
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
        model.to(self.device)
        model.eval()
        
        # Test on some files
        test_dir = Path("data/train")
        test_files_list = list(test_dir.glob("*.pdf"))[:test_files]
        
        total_accuracy = 0.0
        total_fields = 0
        
        for pdf_file in test_files_list:
            json_file = pdf_file.with_suffix(".json")
            if not json_file.exists():
                continue
            
            try:
                # Load ground truth
                with open(json_file, 'r', encoding='utf-8') as f:
                    gt = json.load(f)
                
                # Load image
                image = self.load_image(str(pdf_file))
                
                # Prepare input
                pixel_values = processor(image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate text
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=512,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                    )
                
                # Decode generated text
                generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
                
                # Parse and compare
                predicted = self.parser.parse_raw_output(generated_text)
                
                # Calculate field accuracy
                fields_to_check = [
                    ("document_number", gt.get("document_number", "")),
                    ("issue_date", gt.get("issue_date", "")),
                    ("sale_date", gt.get("sale_date", "")),
                    ("buyer_name", gt.get("buyer", {}).get("name", "")),
                    ("seller_name", gt.get("seller", {}).get("name", "")),
                ]
                
                for field_name, gt_value in fields_to_check:
                    if gt_value:
                        pred_value = predicted.get(field_name, "")
                        if self.normalize_text(pred_value) == self.normalize_text(gt_value):
                            total_accuracy += 1.0
                        total_fields += 1
                
            except Exception as e:
                print(f"⚠️ Error evaluating {pdf_file.name}: {e}")
        
        accuracy = total_accuracy / total_fields if total_fields > 0 else 0.0
        print(f"📊 Retrained model accuracy: {accuracy:.4f}")
        
        return {"accuracy": accuracy, "test_files": len(test_files_list)}

def main():
    """Main retraining function"""
    print("🚀 Invoice Model Retraining")
    print("=" * 50)
    
    # Initialize retrainer
    retrainer = InvoiceRetrainer()
    
    # Check if retrain data exists
    retrain_dir = Path("data/retrain")
    if not retrain_dir.exists():
        print("❌ No retrain data found! Please create data/retrain/ directory with PDF and JSON files.")
        return
    
    # Retrain model
    checkpoint_name = retrainer.retrain_model(epochs=3, learning_rate=3e-5)
    
    if checkpoint_name:
        # Evaluate retrained model
        evaluation = retrainer.evaluate_retrained_model(checkpoint_name)
        
        print(f"\n🎯 Retraining completed!")
        print(f"New checkpoint: {checkpoint_name}")
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
        
        if evaluation['accuracy'] >= 0.90:
            print("🎉 SUCCESS! Retrained model reached 90% accuracy!")
        else:
            print("📈 Good progress, but more training may be needed.")

if __name__ == "__main__":
    main() 