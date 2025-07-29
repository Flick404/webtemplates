import json
import torch
from pathlib import Path
from transformers import DonutProcessor, VisionEncoderDecoderModel
import shutil
import datetime

def save_current_checkpoint():
    """Save the current training checkpoint"""
    print("💾 Saving current training checkpoint...")
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoint_{timestamp}"
    
    # Check if the model directory exists
    model_dir = Path("./donut_invoice_model")
    if not model_dir.exists():
        print("❌ No model directory found!")
        return
    
    try:
        # Copy the entire model directory
        shutil.copytree(model_dir, checkpoint_dir)
        print(f"✅ Checkpoint saved to: {checkpoint_dir}")
        
        # Also save training progress if it exists
        progress_file = Path("outputs/training_progress_v2.json")
        if progress_file.exists():
            shutil.copy2(progress_file, f"{checkpoint_dir}/training_progress.json")
            print(f"✅ Training progress saved to: {checkpoint_dir}/training_progress.json")
        
        # Create a summary file
        summary = {
            "checkpoint_time": timestamp,
            "checkpoint_dir": checkpoint_dir,
            "model_files": list(model_dir.glob("*")),
            "note": "Current training checkpoint saved"
        }
        
        with open(f"{checkpoint_dir}/checkpoint_info.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Checkpoint info saved to: {checkpoint_dir}/checkpoint_info.json")
        
        return checkpoint_dir
        
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        return None

def list_checkpoints():
    """List all available checkpoints"""
    print("📋 Available checkpoints:")
    
    checkpoints = list(Path(".").glob("checkpoint_*"))
    if not checkpoints:
        print("No checkpoints found")
        return
    
    for checkpoint in sorted(checkpoints):
        if checkpoint.is_dir():
            print(f"  - {checkpoint.name}")

if __name__ == "__main__":
    print("🚀 Save Current Training Checkpoint")
    print("=" * 40)
    
    # Save current checkpoint
    checkpoint_dir = save_current_checkpoint()
    
    if checkpoint_dir:
        print(f"\n🎉 Checkpoint saved successfully!")
        print(f"Location: {checkpoint_dir}")
        
        # List all checkpoints
        print("\n" + "=" * 40)
        list_checkpoints()
    else:
        print("\n❌ Failed to save checkpoint") 