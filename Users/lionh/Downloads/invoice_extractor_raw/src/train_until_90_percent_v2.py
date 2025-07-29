import json
import os
import torch
from pathlib import Path
from PIL import Image
import fitz
import io
from transformers import DonutProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer
from datasets import Dataset
import re
import time

class DonutTrainer90PercentV2:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Donut model and processor
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        # Move to device
        self.model.to(self.device)
        
        # Set decoder start token
        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Set max length
        self.model.config.max_length = 512
        
        self.training_history = []
        
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
        
        return "\n".join(lines)
    
    def prepare_dataset(self, data_dir: str = "data/train") -> Dataset:
        """Prepare dataset for Donut training"""
        print("🔄 Preparing Donut dataset...")
        
        data_dir = Path(data_dir)
        pdf_files = list(data_dir.glob("*.pdf"))
        
        dataset_data = []
        
        for pdf_file in pdf_files:
            json_file = pdf_file.with_suffix(".json")
            
            if not json_file.exists():
                continue
            
            try:
                # Load ground truth
                with open(json_file, 'r', encoding='utf-8') as f:
                    gt = json.load(f)
                
                # Load image
                image = self.load_image(str(pdf_file))
                
                # Format ground truth
                target_text = self.format_ground_truth(gt)
                
                if target_text.strip():
                    dataset_data.append({
                        "image": image,
                        "text": target_text
                    })
                
            except Exception as e:
                print(f"⚠️ Error processing {pdf_file.name}: {e}")
        
        print(f"✅ Prepared {len(dataset_data)} samples for training")
        
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
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        # Remove extra spaces and convert to lowercase
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def calculate_similarity(self, pred: str, gt: str) -> float:
        """Calculate similarity between predicted and ground truth text"""
        pred_norm = self.normalize_text(pred)
        gt_norm = self.normalize_text(gt)
        
        if not gt_norm:
            return 1.0 if not pred_norm else 0.0
        
        if not pred_norm:
            return 0.0
        
        # Exact match
        if pred_norm == gt_norm:
            return 1.0
        
        # Check if one contains the other
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return 0.9
        
        # Word overlap
        pred_words = set(pred_norm.split())
        gt_words = set(gt_norm.split())
        
        if not gt_words:
            return 0.0
        
        intersection = pred_words.intersection(gt_words)
        union = pred_words.union(gt_words)
        
        if not union:
            return 0.0
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union)
        
        # Also consider word-level accuracy
        word_accuracy = len(intersection) / len(gt_words)
        
        # Return the higher of the two
        return max(jaccard, word_accuracy)
    
    def evaluate_accuracy(self, model_path: str = "./donut_invoice_model") -> float:
        """Evaluate current model accuracy with better metrics"""
        print("🔍 Evaluating current model accuracy...")
        
        try:
            # Load current model
            processor = DonutProcessor.from_pretrained(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
        except:
            print("⚠️ Could not load model, using base model")
            processor = self.processor
            model = self.model
        
        data_dir = Path("data/train")
        pdf_files = list(data_dir.glob("*.pdf"))
        
        total_similarity = 0.0
        total_samples = 0
        
        print("Testing on first 5 files...")
        
        for pdf_file in pdf_files[:5]:  # Test on first 5 files
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
                
                # Get ground truth text
                gt_text = self.format_ground_truth(gt)
                
                # Calculate similarity
                similarity = self.calculate_similarity(generated_text, gt_text)
                total_similarity += similarity
                total_samples += 1
                
                print(f"  {pdf_file.name}: {similarity:.3f}")
                print(f"    GT: {gt_text[:100]}...")
                print(f"    PRED: {generated_text[:100]}...")
                
            except Exception as e:
                print(f"⚠️ Error evaluating {pdf_file.name}: {e}")
        
        accuracy = total_similarity / total_samples if total_samples > 0 else 0.0
        print(f"📊 Current accuracy: {accuracy:.4f}")
        return accuracy
    
    def train_epoch(self, dataset: Dataset, epochs: int = 3) -> float:
        """Train for a few epochs and return accuracy"""
        print(f"🚀 Training for {epochs} epochs...")
        
        # Preprocess dataset
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        train_dataset = processed_dataset.select(range(int(0.8 * len(processed_dataset))))
        eval_dataset = processed_dataset.select(range(int(0.8 * len(processed_dataset)), len(processed_dataset)))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./donut_invoice_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=20,
            save_steps=50,
            logging_steps=5,
            learning_rate=5e-5,
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
        trainer.train()
        
        # Save model
        trainer.save_model("./donut_invoice_model")
        self.processor.save_pretrained("./donut_invoice_model")
        
        # Evaluate accuracy
        accuracy = self.evaluate_accuracy()
        return accuracy
    
    def train_until_90_percent(self, max_iterations: int = 10):
        """Train until we reach 90% accuracy or max iterations"""
        print("🎯 Training until 90% accuracy (V2)...")
        print("=" * 50)
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        iteration = 0
        current_accuracy = 0.0
        
        while current_accuracy < 0.90 and iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            print(f"Current accuracy: {current_accuracy:.4f}")
            
            # Train for a few epochs
            epochs_per_iteration = 3 if iteration <= 3 else 2
            current_accuracy = self.train_epoch(dataset, epochs=epochs_per_iteration)
            
            # Record history
            self.training_history.append({
                "iteration": iteration,
                "accuracy": current_accuracy,
                "epochs": epochs_per_iteration
            })
            
            print(f"✅ Iteration {iteration} completed. Accuracy: {current_accuracy:.4f}")
            
            # Save progress
            with open("outputs/training_progress_v2.json", "w") as f:
                json.dump(self.training_history, f, indent=2)
            
            if current_accuracy >= 0.90:
                print(f"🎉 Target accuracy reached! Final accuracy: {current_accuracy:.4f}")
                break
        
        if current_accuracy < 0.90:
            print(f"⚠️ Max iterations reached. Final accuracy: {current_accuracy:.4f}")
        
        # Save final model
        print("💾 Saving final model...")
        self.model.save_pretrained("./donut_invoice_model_final_v2")
        self.processor.save_pretrained("./donut_invoice_model_final_v2")
        
        return current_accuracy

def main():
    """Main training function"""
    print("🚀 Donut Training Until 90% Accuracy (V2)")
    print("=" * 50)
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = DonutTrainer90PercentV2()
    
    # Train until 90% accuracy
    final_accuracy = trainer.train_until_90_percent(max_iterations=10)
    
    print(f"\n🎯 Training completed!")
    print(f"Final accuracy: {final_accuracy:.4f}")
    
    if final_accuracy >= 0.90:
        print("🎉 SUCCESS! Reached 90% accuracy!")
    else:
        print("⚠️ Did not reach 90% accuracy, but training completed.")

if __name__ == "__main__":
    main() 