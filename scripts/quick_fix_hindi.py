"""
Quick Fix for Hindi Data Quality
Filters out chunks < 200 characters
Run time: ~2 minutes
"""

import json

def quick_fix_hindi():
    """
    Quick fix: Filter out bad Hindi chunks
    """
    input_file = "data/processed/hi_chunks.jsonl"
    output_file = "data/processed/hi_chunks_filtered.jsonl"
    
    good_chunks = 0
    bad_chunks = 0
    
    print("Filtering Hindi chunks...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line)
                text = data.get('text', '')
                
                # Filter: Keep only chunks >= 200 chars
                if len(text) >= 200:
                    f_out.write(line)
                    good_chunks += 1
                else:
                    bad_chunks += 1
                    
                # Progress
                if (good_chunks + bad_chunks) % 1000 == 0:
                    print(f"Processed: {good_chunks + bad_chunks} chunks...")
    
    print()
    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"✅ Good chunks (>= 200 chars): {good_chunks}")
    print(f"❌ Bad chunks filtered (< 200 chars): {bad_chunks}")
    print(f"📊 Percentage kept: {good_chunks/(good_chunks+bad_chunks)*100:.1f}%")
    print()
    print(f"New file created: {output_file}")
    print()
    print("NEXT STEPS:")
    print("1. Backup old file:")
    print("   copy data\\processed\\hi_chunks.jsonl data\\processed\\hi_chunks_backup.jsonl")
    print()
    print("2. Replace with filtered file:")
    print("   copy data\\processed\\hi_chunks_filtered.jsonl data\\processed\\hi_chunks.jsonl")
    print()
    print("3. Rebuild FAISS index:")
    print("   python run_phase2.py")

if __name__ == "__main__":
    quick_fix_hindi()
