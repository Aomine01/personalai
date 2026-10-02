#!/usr/bin/env python3
import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from config import settings
from generators.xlsx_gen import generate_excel
from utils.cleanup import sweep_tmp_dir

async def handle_excel(prompt: str):
    print(f"Generating Excel spreadsheet for prompt: '{prompt}'...")
    try:
        file_path, _, _, _ = await generate_excel(prompt)
        print(f"SUCCESS: {file_path}")
    except Exception as e:
        print(f"ERROR: Failed to generate Excel. Details: {e}", file=sys.stderr)
        sys.exit(1)

def handle_cleanup():
    print("Sweeping temporary files...")
    sweep_tmp_dir()
    print("SUCCESS: Temp files cleared.")

def main():
    parser = argparse.ArgumentParser(description="CLI interface for Telegram AI Agent utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Excel generation sub-command
    excel_parser = subparsers.add_parser("excel", help="Generate an Excel spreadsheet from a prompt.")
    excel_parser.add_argument("prompt", type=str, help="The prompt/data description to generate a table for.")

    # Cleanup sub-command
    subparsers.add_parser("cleanup", help="Clean up temporary files.")

    args = parser.parse_args()

    if args.command == "excel":
        asyncio.run(handle_excel(args.prompt))
    elif args.command == "cleanup":
        handle_cleanup()

if __name__ == "__main__":
    main()
