#!/usr/bin/env python3
"""One-off script to rename existing log files and update internal references
to match the new method naming convention."""

import os
import json
import shutil

LOGS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# Order matters: longer/more-specific patterns first to avoid partial replacements
RENAME_MAP = [
    ('FlexLoRA_SVD_A', 'FeDoRA'),
    ('FlexLoRA_FreezeA', 'FL+DoRA(FFALoRA)'),
    ('LoRA_FlexLoRA', 'FlexLoRA'),
    ('LoRA_TrainableA', 'FedIT'),
    ('FFA-LORA', 'FFA-LoRA'),
    # FlexLoRA must come LAST (after FlexLoRA_SVD_A, FlexLoRA_FreezeA, LoRA_FlexLoRA)
    # But we need to be careful: at this point, 'FlexLoRA_SVD_A' is already 'FeDoRA',
    # 'FlexLoRA_FreezeA' is already 'FL+DoRA(FFALoRA)',
    # 'LoRA_FlexLoRA' is already 'FlexLoRA'
    # So standalone 'FlexLoRA' in filenames (from run_roberta_glue_experiments) should become 'FL+DoRA'
    # But now 'FlexLoRA' could also match the just-renamed 'FlexLoRA' (from LoRA_FlexLoRA→FlexLoRA).
    # To handle this correctly, we process file by file, applying all renames to the ORIGINAL name.
]

def apply_renames(original_name):
    """Apply all renames to an original filename, respecting order."""
    name = original_name
    for old, new in RENAME_MAP:
        name = name.replace(old, new)
    
    # Now handle the standalone 'FlexLoRA' → 'FL+DoRA' rename
    # This should only match 'FlexLoRA' that was originally 'FlexLoRA' (from glue experiments)
    # not the 'FlexLoRA' that was just created from 'LoRA_FlexLoRA'
    # Since LoRA_FlexLoRA was already renamed above, the remaining 'FlexLoRA' in the name
    # would be the original standalone one from roberta_glue_experiments
    return name


def rename_in_directory(dirpath):
    """Process all files in a directory."""
    if not os.path.exists(dirpath):
        return
    
    for root, dirs, files in os.walk(dirpath):
        # Process files
        for fname in sorted(files):
            old_path = os.path.join(root, fname)
            new_fname = apply_renames(fname)
            new_path = os.path.join(root, new_fname)
            
            if old_path != new_path:
                print(f"  RENAME: {fname} -> {new_fname}")
                shutil.move(old_path, new_path)
            
            # Update JSON contents if applicable
            if new_path.endswith('.json'):
                try:
                    with open(new_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'log_file_name' in data:
                        old_lfn = data['log_file_name']
                        new_lfn = apply_renames(old_lfn)
                        if old_lfn != new_lfn:
                            data['log_file_name'] = new_lfn
                            with open(new_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=4, ensure_ascii=False)
                            print(f"  UPDATE JSON: log_file_name '{old_lfn}' -> '{new_lfn}'")
                except Exception as e:
                    print(f"  WARNING: Could not update JSON {new_path}: {e}")


def main():
    # However, the standalone 'FlexLoRA' rename is tricky.
    # Files in roberta_glue_experiments have names like 'glue_sst2_FlexLoRA.log'
    # Files in roberta_lora_experiments have names like 'glue_sst2_LoRA_FlexLoRA.log'
    # After LoRA_FlexLoRA -> FlexLoRA, the lora_experiments file becomes 'glue_sst2_FlexLoRA.log'
    # which would CONFLICT with the original 'glue_sst2_FlexLoRA.log' in glue_experiments.
    # But they're in DIFFERENT directories so no conflict.
    
    # We need to handle roberta_glue_experiments separately:
    # In roberta_glue_experiments, 'FlexLoRA' (standalone, from --peft dora) should become 'FL+DoRA'
    # In roberta_lora_experiments, 'LoRA_FlexLoRA' should become 'FlexLoRA'
    
    print("=" * 60)
    print("Renaming log files to new method names")
    print("=" * 60)
    
    # Process roberta_glue_experiments first
    glue_dir = os.path.join(LOGS_ROOT, 'roberta_glue_experiments')
    print(f"\n--- Processing: {glue_dir} ---")
    if os.path.exists(glue_dir):
        for root, dirs, files in os.walk(glue_dir):
            for fname in sorted(files):
                old_path = os.path.join(root, fname)
                new_fname = fname
                # Apply specific renames for glue experiments
                for old, new in RENAME_MAP:
                    new_fname = new_fname.replace(old, new)
                # Now handle standalone FlexLoRA -> FL+DoRA
                # At this point FlexLoRA_SVD_A and FlexLoRA_FreezeA are already renamed
                # So any remaining 'FlexLoRA' is the standalone one
                new_fname = new_fname.replace('FlexLoRA', 'FL+DoRA')
                
                new_path = os.path.join(root, new_fname)
                if old_path != new_path:
                    print(f"  RENAME: {fname} -> {new_fname}")
                    shutil.move(old_path, new_path)
                
                if new_path.endswith('.json'):
                    try:
                        with open(new_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if 'log_file_name' in data:
                            old_lfn = data['log_file_name']
                            new_lfn = old_lfn
                            for old, new in RENAME_MAP:
                                new_lfn = new_lfn.replace(old, new)
                            new_lfn = new_lfn.replace('FlexLoRA', 'FL+DoRA')
                            if old_lfn != new_lfn:
                                data['log_file_name'] = new_lfn
                                with open(new_path, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, indent=4, ensure_ascii=False)
                                print(f"  UPDATE JSON: log_file_name '{old_lfn}' -> '{new_lfn}'")
                    except Exception as e:
                        print(f"  WARNING: Could not update JSON {new_path}: {e}")
    
    # Process roberta_lora_experiments
    lora_dir = os.path.join(LOGS_ROOT, 'roberta_lora_experiments')
    print(f"\n--- Processing: {lora_dir} ---")
    if os.path.exists(lora_dir):
        for root, dirs, files in os.walk(lora_dir):
            for fname in sorted(files):
                old_path = os.path.join(root, fname)
                new_fname = fname
                # LoRA_FlexLoRA -> FlexLoRA
                new_fname = new_fname.replace('LoRA_FlexLoRA', 'FlexLoRA')
                # FFA-LORA -> FFA-LoRA
                new_fname = new_fname.replace('FFA-LORA', 'FFA-LoRA')
                # FedEx-LoRA stays
                
                new_path = os.path.join(root, new_fname)
                if old_path != new_path:
                    print(f"  RENAME: {fname} -> {new_fname}")
                    shutil.move(old_path, new_path)
                
                if new_path.endswith('.json'):
                    try:
                        with open(new_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if 'log_file_name' in data:
                            old_lfn = data['log_file_name']
                            new_lfn = old_lfn
                            new_lfn = new_lfn.replace('LoRA_FlexLoRA', 'FlexLoRA')
                            new_lfn = new_lfn.replace('FFA-LORA', 'FFA-LoRA')
                            if old_lfn != new_lfn:
                                data['log_file_name'] = new_lfn
                                with open(new_path, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, indent=4, ensure_ascii=False)
                                print(f"  UPDATE JSON: log_file_name '{old_lfn}' -> '{new_lfn}'")
                    except Exception as e:
                        print(f"  WARNING: Could not update JSON {new_path}: {e}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
