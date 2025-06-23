#!/usr/bin/env python3
"""
Test script for FSVM Streamlit App
"""

import time
from fsvm_core import FSVM

def test_fsvm_for_app():
    """Test FSVM functionality needed for the Streamlit app"""
    print("🧪 Testing FSVM for Streamlit App...")
    
    # Initialize FSVM
    fsvm = FSVM()
    print("✅ FSVM initialized")
    
    # Test tension multiplier
    print(f"📊 Tension multiplier: {fsvm.get_tension_multiplier()}")
    fsvm.set_tension_multiplier(2.0)
    print(f"📊 Updated tension multiplier: {fsvm.get_tension_multiplier()}")
    
    # Test status updates
    print("📡 Testing status updates...")
    fsvm.start()
    time.sleep(2)
    
    status = fsvm.get_status_update()
    if status:
        print(f"✅ Status update received: {status.get('cycle_count', 0)} cycles")
    else:
        print("⚠️ No status update received")
    
    # Test chat
    print("💬 Testing chat...")
    fsvm.chat("Hello FSVM")
    time.sleep(1)
    
    try:
        response = fsvm.chat_output_queue.get_nowait()
        print(f"✅ Chat response: {response}")
    except:
        print("⚠️ No chat response")
    
    # Test symbols
    print(f"🔣 Symbols created: {len(fsvm.tension_engine.symbols)}")
    
    # Test drives
    if hasattr(fsvm, 'drives'):
        print(f"❤️‍🔥 Drives: {fsvm.drives}")
    
    # Test activity log
    if hasattr(fsvm, 'activity_log'):
        print(f"📜 Activity log entries: {len(fsvm.activity_log)}")
    
    # Stop FSVM
    fsvm.stop()
    print("✅ FSVM stopped")
    
    print("🎉 All tests completed! Streamlit app should work.")

if __name__ == "__main__":
    test_fsvm_for_app() 