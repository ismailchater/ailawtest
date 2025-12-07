"""
Document Sync Script for Qdrant.
Run this script to sync PDF files from module folders to Qdrant.

Usage:
    python sync_documents.py                    # Sync all modules
    python sync_documents.py --module cgi       # Sync specific module
    python sync_documents.py --module cgi --clear  # Clear and re-sync module
    python sync_documents.py --status           # Show sync status
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODULES, get_module_config
from document_loader import create_folder_processor
from vector_store import create_vector_store_manager


def get_enabled_modules() -> Dict[str, Dict[str, Any]]:
    """Get all enabled modules."""
    return {k: v for k, v in MODULES.items() if v.get("enabled", False)}


def sync_module(module_id: str, clear_first: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Sync a single module's documents to Qdrant.
    
    Args:
        module_id: ID of the module to sync
        clear_first: Whether to clear existing vectors first
        verbose: Whether to print progress
        
    Returns:
        Dict with sync results
    """
    module_config = get_module_config(module_id)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Syncing module: {module_config['name']} ({module_id})")
        print(f"{'='*60}")
    
    # Initialize processors
    doc_processor = create_folder_processor(module_config)
    vector_manager = create_vector_store_manager(module_config)
    
    # Ensure folder exists
    doc_processor.ensure_folder_exists()
    
    # Get PDF files
    pdf_files = doc_processor.get_pdf_files()
    
    if not pdf_files:
        if verbose:
            print(f"  No PDF files found in: {doc_processor.folder_path}")
        return {
            "module_id": module_id,
            "files_processed": 0,
            "chunks_added": 0,
            "errors": [],
            "success": True
        }
    
    if verbose:
        print(f"  Found {len(pdf_files)} PDF file(s)")
    
    # Clear existing vectors if requested
    if clear_first:
        if verbose:
            print("  Clearing existing vectors...")
        vector_manager.clear_collection()
    
    # Process files
    total_chunks = 0
    errors = []
    
    for result in doc_processor.process_all_files():
        file_name = result["file_name"]
        
        if result["success"]:
            chunks = result["chunks"]
            chunk_count = len(chunks)
            
            if verbose:
                print(f"  Processing: {file_name} ({chunk_count} chunks)")
            
            # Add to Qdrant
            if chunks:
                vector_manager.add_documents(chunks)
                total_chunks += chunk_count
        else:
            error_msg = f"{file_name}: {result['error']}"
            errors.append(error_msg)
            if verbose:
                print(f"  ERROR: {error_msg}")
    
    if verbose:
        print(f"\n  Total chunks added: {total_chunks}")
        if errors:
            print(f"  Errors: {len(errors)}")
    
    return {
        "module_id": module_id,
        "files_processed": len(pdf_files),
        "chunks_added": total_chunks,
        "errors": errors,
        "success": len(errors) == 0
    }


def sync_all_modules(clear_first: bool = False) -> List[Dict[str, Any]]:
    """Sync all enabled modules."""
    results = []
    enabled_modules = get_enabled_modules()
    
    print(f"\nSyncing {len(enabled_modules)} enabled module(s)...")
    
    for module_id in enabled_modules:
        result = sync_module(module_id, clear_first=clear_first)
        results.append(result)
    
    return results


def show_status():
    """Show sync status for all modules."""
    print("\n" + "="*60)
    print("DOCUMENT SYNC STATUS")
    print("="*60)
    
    for module_id, module_config in MODULES.items():
        enabled = module_config.get("enabled", False)
        status = "ENABLED" if enabled else "DISABLED"
        
        print(f"\n{module_config['icon']} {module_config['name']} ({module_id}) - {status}")
        
        if enabled:
            # Check folder
            folder_path = Path(module_config["documents_folder"])
            if folder_path.exists():
                pdf_count = len(list(folder_path.glob("*.pdf")))
                print(f"   Folder: {folder_path} ({pdf_count} PDFs)")
            else:
                print(f"   Folder: {folder_path} (NOT FOUND)")
            
            # Check Qdrant collection
            try:
                vector_manager = create_vector_store_manager(module_config)
                info = vector_manager.get_collection_info()
                if info["exists"]:
                    print(f"   Qdrant: {info['count']} vectors")
                else:
                    print(f"   Qdrant: Collection not created")
            except Exception as e:
                print(f"   Qdrant: Error - {str(e)}")


def sync_single_file(module_id: str, file_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Sync a single file to Qdrant (useful for incremental updates).
    
    Args:
        module_id: ID of the module
        file_path: Path to the PDF file
        verbose: Whether to print progress
    """
    from pathlib import Path
    from document_loader import FolderDocumentProcessor
    
    module_config = get_module_config(module_id)
    vector_manager = create_vector_store_manager(module_config)
    
    pdf_path = Path(file_path)
    
    if not pdf_path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}
    
    if verbose:
        print(f"Processing: {pdf_path.name}")
    
    # Delete existing vectors for this file
    vector_manager.delete_by_file(pdf_path.name)
    
    # Process file
    processor = FolderDocumentProcessor(
        folder_path=pdf_path.parent,
        module_id=module_id
    )
    
    try:
        chunks = processor.process_single_file(pdf_path)
        vector_manager.add_documents(chunks)
        
        if verbose:
            print(f"  Added {len(chunks)} chunks")
        
        return {
            "success": True,
            "file_name": pdf_path.name,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Sync documents to Qdrant")
    parser.add_argument("--module", "-m", help="Specific module to sync")
    parser.add_argument("--clear", "-c", action="store_true", help="Clear existing vectors before sync")
    parser.add_argument("--status", "-s", action="store_true", help="Show sync status")
    parser.add_argument("--file", "-f", help="Sync a single file (requires --module)")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    if args.file:
        if not args.module:
            print("ERROR: --file requires --module")
            sys.exit(1)
        result = sync_single_file(args.module, args.file)
        if not result["success"]:
            print(f"ERROR: {result.get('error')}")
            sys.exit(1)
        return
    
    if args.module:
        result = sync_module(args.module, clear_first=args.clear)
        if not result["success"]:
            print(f"\nCompleted with {len(result['errors'])} error(s)")
            sys.exit(1)
    else:
        results = sync_all_modules(clear_first=args.clear)
        errors = sum(1 for r in results if not r["success"])
        if errors:
            print(f"\nCompleted with errors in {errors} module(s)")
            sys.exit(1)
    
    print("\n✅ Sync completed successfully!")


if __name__ == "__main__":
    main()
