#!/usr/bin/env python3
import os
import argparse

def collect_files(extensions, output_filename):
    """
    Recursively search for files with the given extensions starting from the current directory,
    and write their contents into the output file with headers.
    """
    # Normalize extensions to ensure each starts with a dot.
    normalized_exts = [ext if ext.startswith('.') else '.' + ext for ext in extensions]

    with open(output_filename, "w", encoding="utf-8") as outfile:
        # Walk through the current directory and subdirectories.
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(tuple(normalized_exts)):
                    full_path = os.path.join(root, file)
                    header = f"===== Start of file: {full_path} =====\n"
                    footer = f"\n===== End of file: {full_path} =====\n\n"
                    
                    outfile.write(header)
                    
                    try:
                        with open(full_path, "r", encoding="utf-8") as infile:
                            content = infile.read()
                            outfile.write(content)
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}\n")
                    
                    outfile.write(footer)

def main():
    parser = argparse.ArgumentParser(
        description="Recursively collect all files with the given extensions into one text file."
    )
    parser.add_argument(
        "extensions",
        type=str,
        nargs="+",
        help="One or more file extensions to search for (e.g., py js txt for '.py', '.js', '.txt' files)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="all_files.txt",
        help="Name of the output file (default: all_files.txt)."
    )
    
    args = parser.parse_args()
    
    collect_files(args.extensions, args.output)
    print(f"Files with extensions {args.extensions} have been collected into '{args.output}'")

if __name__ == "__main__":
    main()
