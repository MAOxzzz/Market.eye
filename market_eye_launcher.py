import subprocess
import threading
import webbrowser
import time
import os
import sys
import signal
import socket
import requests
from urllib.error import URLError
import traceback
import shutil

def start_backend():
    """Start the FastAPI backend server"""
    print("Starting FastAPI backend server...")
    try:
        # When running as executable, use full path to main.py
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            main_path = os.path.join(sys._MEIPASS, "main.py")
            print(f"Using bundled main.py at: {main_path}")
            # Run uvicorn with explicit path
            backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload", "--log-level", "debug"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered output
                cwd=sys._MEIPASS  # Set working directory explicitly
            )
        else:
            # Running as script - use normal path
            backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        return backend_process
    except Exception as e:
        print(f"Error starting backend: {str(e)}")
        traceback.print_exc()
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("Starting Streamlit frontend...")
    try:
        # When running as executable, use full path to streamlit_app.py
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            streamlit_path = os.path.join(sys._MEIPASS, "streamlit_app.py")
            print(f"Using bundled streamlit_app.py at: {streamlit_path}")
            # Run streamlit with explicit path
            frontend_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", streamlit_path, "--browser.serverAddress=localhost", "--server.port=8501"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered output
                cwd=sys._MEIPASS  # Set working directory explicitly
            )
        else:
            # Running as script - use normal path
            frontend_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        return frontend_process
    except Exception as e:
        print(f"Error starting frontend: {str(e)}")
        traceback.print_exc()
        return None

def log_output(process, name):
    """Log the output from a process"""
    try:
        if not process or not process.stdout:
            print(f"No output stream available for {name}")
            return

        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name}] {line.strip()}")
            else:
                # Add a small sleep to prevent CPU spinning
                time.sleep(0.01)
    except Exception as e:
        print(f"Error in log_output for {name}: {str(e)}")
        traceback.print_exc()

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_server(url, timeout=60):
    """Wait for a server to respond"""
    print(f"Waiting for server at {url} to be available...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 400:
                print(f"Server at {url} is ready!")
                return True
        except requests.RequestException:
            # Add a message to show progress
            if int(time.time() - start_time) % 5 == 0:
                print(f"Still waiting for {url}... ({int(time.time() - start_time)}s)")
            time.sleep(1)
    print(f"Timed out waiting for {url} after {timeout} seconds")
    return False

def open_browser():
    """Open browser to the Streamlit interface only when servers are ready"""
    print("Checking if servers are ready...")
    
    # Wait for backend to be available - try the root URL first
    backend_ready = wait_for_server('http://localhost:8000/')
    if not backend_ready:
        print("Warning: Backend server not responding at root URL, trying docs...")
        backend_ready = wait_for_server('http://localhost:8000/docs')
    
    if not backend_ready:
        print("Warning: Backend server not responding within timeout")
        return  # Don't continue if backend isn't ready
    
    # Wait for frontend to be available
    frontend_ready = wait_for_server('http://localhost:8501')
    if frontend_ready:
        print("Opening browser to Streamlit interface...")
        # Use a small delay to ensure server is fully ready
        time.sleep(2)
        webbrowser.open('http://localhost:8501')
    else:
        print("Warning: Streamlit server not responding within timeout")

def handle_shutdown(backend_process, frontend_process, signum=None, frame=None):
    """Gracefully shut down all processes"""
    print("\nShutting down Market Eye AI...")
    
    try:
        if frontend_process and frontend_process.poll() is None:
            print("Stopping Streamlit frontend...")
            try:
                frontend_process.terminate()
                frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Frontend didn't terminate gracefully, forcing kill...")
                frontend_process.kill()
            except Exception as e:
                print(f"Error stopping frontend: {str(e)}")
        
        if backend_process and backend_process.poll() is None:
            print("Stopping FastAPI backend...")
            try:
                backend_process.terminate()
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Backend didn't terminate gracefully, forcing kill...")
                backend_process.kill()
            except Exception as e:
                print(f"Error stopping backend: {str(e)}")
    except KeyboardInterrupt:
        print("Received keyboard interrupt during shutdown")
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")
        traceback.print_exc()
    
    print("Market Eye AI has been shut down.")
    sys.exit(0)

def main():
    """Main function to run the application"""
    print("=" * 50)
    print("Market Eye AI - Starting Application")
    print("=" * 50)
    
    # Check if ports are already in use
    if is_port_in_use(8000):
        print("Error: Port 8000 is already in use. Cannot start backend server.")
        print("Please close any applications using port 8000 and try again.")
        input("Press Enter to exit...")
        return
        
    if is_port_in_use(8501):
        print("Error: Port 8501 is already in use. Cannot start Streamlit server.")
        print("Please close any applications using port 8501 and try again.")
        input("Press Enter to exit...")
        return
    
    # When running as a PyInstaller bundle, the path is different
    # Check if we're running as a bundled app
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as compiled executable - files should be in the _MEIPASS directory
        bundle_dir = sys._MEIPASS
        print(f"Running as bundled application from: {bundle_dir}")
        # Ensure the current directory is the bundle directory
        os.chdir(bundle_dir)
        
        # Check if necessary bundled files exist
        if not os.path.exists(os.path.join(bundle_dir, "main.py")):
            print(f"Error: main.py not found in {bundle_dir}")
            print("The executable may be corrupted. Please rebuild it.")
            input("Press Enter to exit...")
            return
        
        if not os.path.exists(os.path.join(bundle_dir, "streamlit_app.py")):
            print(f"Error: streamlit_app.py not found in {bundle_dir}")
            print("The executable may be corrupted. Please rebuild it.")
            input("Press Enter to exit...")
            return
    else:
        # Check if necessary files exist when running as a script
        if not os.path.exists("main.py"):
            print("Error: main.py not found. Please run from the project root directory.")
            input("Press Enter to exit...")
            return
        
        if not os.path.exists("streamlit_app.py"):
            print("Error: streamlit_app.py not found. Please run from the project root directory.")
            input("Press Enter to exit...")
            return
    
    # Start the backend server
    backend_process = start_backend()
    if not backend_process:
        print("Failed to start backend server")
        input("Press Enter to exit...")
        return
    
    # Set up a thread to read and log backend output
    backend_log_thread = threading.Thread(
        target=log_output,
        args=(backend_process, "Backend"),
        daemon=True
    )
    backend_log_thread.start()
    
    # Give the backend a moment to start
    time.sleep(3)
    
    # Start the frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("Failed to start frontend server")
        handle_shutdown(backend_process, None)
        input("Press Enter to exit...")
        return
    
    # Set up a thread to read and log frontend output
    frontend_log_thread = threading.Thread(
        target=log_output,
        args=(frontend_process, "Frontend"),
        daemon=True
    )
    frontend_log_thread.start()
    
    # Open browser only when servers are ready
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    print("\nMarket Eye AI is running!")
    print("- Frontend: http://localhost:8501")
    print("- Backend API: http://localhost:8000")
    print("- Backend API docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to shut down.")
    
    try:
        # Keep the main thread alive and monitor child processes
        while True:
            # Check if either process has terminated unexpectedly
            if backend_process.poll() is not None:
                exit_code = backend_process.poll()
                print(f"Backend server has stopped with exit code {exit_code}. Shutting down...")
                handle_shutdown(backend_process, frontend_process)
                break
            
            if frontend_process.poll() is not None:
                exit_code = frontend_process.poll()
                print(f"Frontend server has stopped with exit code {exit_code}. Shutting down...")
                handle_shutdown(backend_process, frontend_process)
                break
            
            # Add a sleep to prevent high CPU usage in this loop
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
        handle_shutdown(backend_process, frontend_process)
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        traceback.print_exc()
        handle_shutdown(backend_process, frontend_process)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1) 