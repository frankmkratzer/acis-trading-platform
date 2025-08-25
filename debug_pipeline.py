#!/usr/bin/env python3
"""
Debug helper for testing individual scripts with proper timeout and monitoring
"""

import subprocess
import sys
import time
import threading
import psutil
import os
from pathlib import Path

def monitor_process(proc, script_name, timeout):
    """Monitor a running process and provide real-time feedback"""
    start_time = time.time()
    
    while proc.poll() is None:
        elapsed = time.time() - start_time
        
        try:
            # Get process info
            ps = psutil.Process(proc.pid)
            cpu_percent = ps.cpu_percent(interval=0.1)
            memory_mb = ps.memory_info().rss / 1024 / 1024
            
            # Check for child processes
            children = ps.children(recursive=True)
            num_children = len(children)
            
            print(f"\r[{elapsed:.1f}s] {script_name} - CPU: {cpu_percent:.1f}% | "
                  f"Memory: {memory_mb:.1f}MB | Children: {num_children} | "
                  f"Timeout in: {timeout - elapsed:.1f}s", end="", flush=True)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        time.sleep(1)
        
        if elapsed > timeout:
            print(f"\n⚠️ Timeout reached! Terminating {script_name}...")
            proc.terminate()
            time.sleep(2)
            if proc.poll() is None:
                proc.kill()
            break
    
    print()  # New line after monitoring

def test_script(script_name, timeout=None, show_output=True):
    """Test a single script with monitoring"""
    script_path = Path(script_name)
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return False
    
    if timeout is None:
        timeout = 300  # 5 minute default
    
    print(f"\n{'='*60}")
    print(f"Testing: {script_name}")
    print(f"Timeout: {timeout} seconds")
    print(f"Path: {script_path.absolute()}")
    print(f"{'='*60}\n")
    
    try:
        # Start the process
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE if not show_output else None,
            stderr=subprocess.PIPE if not show_output else None,
            text=True
        )
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(
            target=monitor_process,
            args=(proc, script_name, timeout)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for completion
        stdout, stderr = proc.communicate(timeout=timeout)
        
        if proc.returncode == 0:
            print(f"✅ {script_name} completed successfully!")
            return True
        else:
            print(f"❌ {script_name} failed with return code: {proc.returncode}")
            if stderr and not show_output:
                print(f"Error output:\n{stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {script_name} timed out after {timeout} seconds")
        proc.kill()
        return False
    except Exception as e:
        print(f"💥 Error running {script_name}: {e}")
        return False

def main():
    """Main test runner"""
    print("🔧 Pipeline Script Debugger")
    print("="*60)
    
    # Scripts to test in order
    test_scripts = [
        ("setup_schema.py", 300),  # 5 minutes
        ("fetch_symbol_metadata.py", 600),  # 10 minutes
        ("fetch_prices.py", 1800),  # 30 minutes
    ]
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        script = sys.argv[1]
        timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        show_output = "--show-output" in sys.argv
        test_script(script, timeout, show_output)
    else:
        print("\nUsage:")
        print("  python debug_pipeline.py <script_name> [timeout] [--show-output]")
        print("\nExample:")
        print("  python debug_pipeline.py setup_schema.py 300 --show-output")
        print("\nOr run quick test of setup_schema.py:")
        print("  python debug_pipeline.py --quick-test")
        
        if "--quick-test" in sys.argv:
            # Quick test with increased timeout
            print("\n🚀 Running quick test of setup_schema.py with 5 minute timeout...")
            test_script("setup_schema.py", 300, show_output=True)
        else:
            print("\nAvailable scripts to test:")
            for script, timeout in test_scripts:
                exists = "✓" if Path(script).exists() else "✗"
                print(f"  [{exists}] {script:<40} (timeout: {timeout}s)")

if __name__ == "__main__":
    main()
