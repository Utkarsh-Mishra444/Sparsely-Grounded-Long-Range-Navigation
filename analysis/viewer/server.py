#!/usr/bin/env python3
"""
Log viewer server for navigation experiment analysis.

Usage:
    python -m analysis.viewer.server [LOG_DIR] [--port PORT]

Examples:
    python -m analysis.viewer.server                    # Serves ./logs/ on port 8000
    python -m analysis.viewer.server /path/to/logs      # Custom log directory
    python -m analysis.viewer.server --port 9000        # Custom port
"""

from http.server import SimpleHTTPRequestHandler, HTTPServer
import os
import sys
import json
import argparse
import shutil
from urllib.parse import parse_qs, urlparse, unquote

# Import deep analyzer from analysis package
try:
    from analysis.stats.crawler import crawl_directory_deep
except ImportError:
    crawl_directory_deep = None  # type: ignore[assignment]


def find_experiment_folders(base_dir='.', max_depth=10):
    """
    Recursively find all experiment folders.
    An experiment folder is one that contains a file starting with 'visited_coordinates_'
    or 'visited_coordinates.json', or has openai_calls/gemini_calls/self_position_calls directories.
    
    Returns list of paths relative to the base_dir.
    """
    if max_depth <= 0:
        return []
    
    experiments = []
    
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        rel_path = os.path.relpath(full_path, '.')
        
        if os.path.isdir(full_path):
            # Check if this directory is an experiment by looking for visited_coordinates files
            try:
                listing = os.listdir(full_path)
            except Exception:
                listing = []
            is_experiment = any(
                f.startswith('visited_coordinates_') or f == 'visited_coordinates.json'
                for f in listing
            )
            
            # Also check for API/self-position directories
            has_api_calls = (
                os.path.exists(os.path.join(full_path, 'openai_calls')) or 
                os.path.exists(os.path.join(full_path, 'gemini_calls')) or
                os.path.exists(os.path.join(full_path, 'self_position_calls'))
            )
            
            if is_experiment or has_api_calls:
                experiments.append(rel_path)
            else:
                # Recursively search subdirectories
                sub_experiments = find_experiment_folders(full_path, max_depth - 1)
                experiments.extend(sub_experiments)
    
    return experiments


def is_experiment_folder(folder_path):
    """
    Check if a folder is an experiment folder.
    """
    if not os.path.isdir(folder_path):
        return False
        
    # Check if directory contains visited_coordinates files
    has_visited_file = any(f.startswith('visited_coordinates_') for f in os.listdir(folder_path))
    
    # Check for API/self-position directories
    has_api_calls = (
        os.path.exists(os.path.join(folder_path, 'openai_calls')) or 
        os.path.exists(os.path.join(folder_path, 'gemini_calls')) or
        os.path.exists(os.path.join(folder_path, 'self_position_calls'))
    )
    
    return has_visited_file or has_api_calls


def is_successful_experiment(folder_path):
    """
    Check if an experiment was successful by looking for the success message in log files.
    Success is determined by finding either "Reached within 50 meters of destination after"
    or "Reached destination polygon!" in any log file.
    """
    if not os.path.isdir(folder_path):
        return False

    # Look for terminal log files (terminal_output_*.log)
    log_files = [f for f in os.listdir(folder_path) if f.startswith('terminal_output_') and f.endswith('.log')]

    for log_file in log_files:
        log_path = os.path.join(folder_path, log_file)
        try:
            # Read the last 50 lines of the file to find the success message
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                last_lines = lines[-50:] if len(lines) > 50 else lines

                if any("Reached within 50 meters of destination after" in line or "Reached destination polygon!" in line for line in last_lines):
                    return True
        except Exception:
            pass

    return False


def get_directory_contents(directory='.'):
    """
    Get folders and experiments in a specific directory.
    Returns a dictionary with 'folders', 'experiments', and 'total_experiments' count.
    'total_experiments' includes experiments in the current directory and all subdirectories.
    """
    if not os.path.isdir(directory):
        return {'folders': [], 'experiments': [], 'total_experiments': 0}
    
    contents = {'folders': [], 'experiments': [], 'total_experiments': 0, 'successful_experiments': 0}
    
    # First get immediate contents
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        rel_path = os.path.relpath(full_path, '.')
        
        if os.path.isdir(full_path):
            if is_experiment_folder(full_path):
                # Check if this experiment was successful
                is_successful = is_successful_experiment(full_path)
                
                contents['experiments'].append({
                    'path': rel_path,
                    'successful': is_successful
                })
                contents['total_experiments'] += 1
                
                if is_successful:
                    contents['successful_experiments'] += 1
            else:
                # Recursively check subdirectories
                sub_contents = get_directory_contents(full_path)
                
                # Determine if this directory is a meta-run (marker file presence)
                is_meta = os.path.exists(os.path.join(full_path, 'is_meta.txt'))
                folder_info = {
                    'path': rel_path,
                    'has_successful': sub_contents['successful_experiments'] > 0,
                    'is_meta': is_meta
                }
                
                contents['folders'].append(folder_info)
                contents['total_experiments'] += sub_contents['total_experiments']
                contents['successful_experiments'] += sub_contents['successful_experiments']
    
    # Sort alphabetically by path
    contents['folders'].sort(key=lambda x: x['path'])
    contents['experiments'].sort(key=lambda x: x['path'])
    
    return contents


class CustomHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable caching to reflect file changes immediately
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def _is_experiments(self, path: str) -> bool:
        return path.rstrip('/') == '/experiments'
    
    def log_message(self, format, *args):
        # Override to reduce verbosity - only log errors
        if args and '404' in str(args[0]):
            super().log_message(format, *args)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Debug endpoint to list directory structure
        if parsed_path.path == '/debug-dirs':
            query = parse_qs(parsed_path.query)
            base_dir = query.get('dir', ['.'])[0]
            
            try:
                debug_info = {}
                if os.path.exists(base_dir):
                    for root, dirs, files in os.walk(base_dir):
                        level = root.replace(base_dir, '').count(os.sep)
                        if level < 3:
                            rel_path = os.path.relpath(root, base_dir)
                            debug_info[rel_path] = {
                                'dirs': dirs[:10],
                                'files': [f for f in files if f.endswith(('.json', '.txt'))][:5]
                            }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(debug_info, indent=2).encode())
                return
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Debug error: {str(e)}".encode())
                return
        
        # Endpoint to list all experiment directories recursively (flat list)
        if self._is_experiments(parsed_path.path):
            query = parse_qs(parsed_path.query)
            base_dir = query.get('dir', ['.'])[0]
            experiments = find_experiment_folders(base_dir)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(experiments).encode())
        
        # Endpoint to get directory contents (folders and experiments)
        elif parsed_path.path == '/directory-contents':
            query = parse_qs(parsed_path.query)
            directory = query.get('dir', ['.'])[0]
            
            # Security check to prevent directory traversal
            normalized_path = os.path.normpath(directory)
            if normalized_path.startswith('..'):
                self.send_response(403)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Access denied: Cannot access parent directories")
                return
                
            contents = get_directory_contents(normalized_path)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(contents).encode())
        
        # Endpoint to list files in experiment directory
        elif parsed_path.path == '/files':
            query = parse_qs(parsed_path.query)
            exp = query.get('exp', [''])[0]
            exp_dir = unquote(exp)
            if '%' in exp_dir:
                exp_dir = unquote(exp_dir)
            exp_dir = exp_dir.replace('\\', os.path.sep).replace('%5C', os.path.sep)
            
            if os.path.exists(exp_dir):
                files = [f for f in os.listdir(exp_dir) if f.startswith('visited_coordinates')]
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(files).encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Directory {exp_dir} not found".encode())

        # Endpoint: deep analysis of a directory (returns JSON)
        elif parsed_path.path == '/deep-analysis':
            query = parse_qs(parsed_path.query)
            directory = query.get('dir', ['.'])[0]
            normalized_path = os.path.normpath(directory)
            if normalized_path.startswith('..'):
                self.send_response(403)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Access denied"}).encode())
                return
            if crawl_directory_deep is None:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Deep analyzer not available"}).encode())
                return
            try:
                report = crawl_directory_deep(normalized_path)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(report).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        
        # Endpoint to list API call files
        elif (
            parsed_path.path.endswith('/openai_calls/') or
            parsed_path.path.endswith('/gemini_calls/') or
            parsed_path.path.endswith('/self_position_calls/')
        ):
            try:
                decoded_path = unquote(parsed_path.path)
                if '%' in decoded_path:
                    decoded_path = unquote(decoded_path)
                
                path_parts = decoded_path.strip('/').split('/')
                if len(path_parts) >= 2:
                    api_type = path_parts[-1]
                    exp_path = '/'.join(path_parts[:-1])
                    
                    exp_path_clean = exp_path.replace('\\', '/').replace('%5C', '/')
                    api_dir = os.path.join(*exp_path_clean.split('/'), api_type)
                    
                    if os.path.exists(api_dir):
                        files = [f for f in os.listdir(api_dir) if f.endswith('.json')]
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(files).encode())
                    else:
                        self.send_response(404)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(f"Directory {api_dir} not found".encode())
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b"Invalid path format")
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Error: {str(e)}".encode())
        
        # Serve static files (like index.html) and JSON files directly
        else:
            if self.path.endswith('.json'):
                try:
                    decoded_path = unquote(self.path)
                    file_path = decoded_path.lstrip('/')
                    file_path = file_path.replace('/', os.path.sep).replace('\\', os.path.sep)
                    
                    if os.path.exists(file_path):
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        with open(file_path, 'rb') as f:
                            self.wfile.write(f.read())
                        return
                except Exception:
                    pass
            
            super().do_GET()
    
    def do_HEAD(self):
        parsed_path = urlparse(self.path)
        if self._is_experiments(parsed_path.path):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
        else:
            super().do_HEAD()


def main():
    """Main entry point for the log viewer server."""
    parser = argparse.ArgumentParser(
        description="Log viewer server for navigation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m analysis.viewer.server                    # Serve ./logs/ on port 8000
    python -m analysis.viewer.server /path/to/logs      # Custom log directory
    python -m analysis.viewer.server --port 9000        # Custom port
        """
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="logs",
        help="Directory containing experiment logs (default: ./logs)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run server on (default: 8000)"
    )
    args = parser.parse_args()
    
    # Resolve log directory path
    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory does not exist: {log_dir}")
        sys.exit(1)
    
    # Change to log directory for serving
    os.chdir(log_dir)
    
    # Copy index.html to log directory if not present
    viewer_dir = os.path.dirname(__file__)
    index_src = os.path.join(viewer_dir, "index.html")
    index_dst = os.path.join(log_dir, "index.html")
    if os.path.exists(index_src) and not os.path.exists(index_dst):
        shutil.copy(index_src, index_dst)
        print(f"Copied index.html to {log_dir}")
    
    server = HTTPServer(('localhost', args.port), CustomHandler)
    print(f"Log Viewer Server")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  Serving logs from: {log_dir}")
    print(f"  Press Ctrl+C to stop")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
