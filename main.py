"""
Main CLI Entry Point for Multilingual QA System
Routes to different phases based on user selection
"""

import sys

def print_menu():
    print("\n" + "="*60)
    print("MULTILINGUAL QA SYSTEM")
    print("="*60)
    print("\n[DATA PREPARATION]")
    print("  1. Phase 1: Stream and chunk corpus data")
    print("  2. Phase 2: Build FAISS index")
    print("\n[TESTING & USAGE]")
    print("  3. Phase 3: Test retrieval only")
    print("  4. Phase 4: Full RAG pipeline (no XAI)")
    print("  5. Phase 5: Full RAG pipeline with XAI")
    print("\n[OTHER]")
    print("  0. Exit")
    print("="*60)

def main():
    while True:
        print_menu()
        choice = input("\nSelect phase (0-5): ").strip()
        
        if choice == "0":
            print("Exiting...")
            sys.exit(0)
        
        elif choice == "1":
            print("\n[PHASE 1: Data Streaming and Chunking]")
            from datasets_loader.corpus_streamer import stream_and_chunk
            stream_and_chunk()
            input("\nPress Enter to continue...")
        
        elif choice == "2":
            print("\n[PHASE 2: Building FAISS Index]")
            from retrieval.build_faiss import build_index
            build_index()
            input("\nPress Enter to continue...")
        
        elif choice == "3":
            print("\n[PHASE 3: Retrieval Testing]")
            import subprocess
            subprocess.run(["python", "run_phase3.py"])
            input("\nPress Enter to continue...")
        
        elif choice == "4":
            print("\n[PHASE 4: Full RAG Pipeline]")
            import subprocess
            subprocess.run(["python", "run_phase4.py"])
            input("\nPress Enter to continue...")
        
        elif choice == "5":
            print("\n[PHASE 5: RAG with XAI]")
            import subprocess
            subprocess.run(["python", "run_phase5.py"])
            input("\nPress Enter to continue...")
        
        else:
            print("Invalid choice. Please select 0-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
