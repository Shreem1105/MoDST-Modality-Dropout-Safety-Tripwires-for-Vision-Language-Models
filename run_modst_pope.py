import torch
import torch.utils.data
from PIL import Image
import json
import argparse
import os
import tqdm
from modst.models.vlm_wrapper import LlavaWrapper
from modst.datasets.pope_loader import PopeLoader
import copy

def run_inference(args):
    print(f"Loading Model: {args.model_id} (4-bit)...")
    model = LlavaWrapper(model_id=args.model_id, load_in_4bit=True)
    
    print(f"Loading Dataset: {args.pope_root}...")
    loader = PopeLoader(args.pope_root, args.images_root)
    dataset = loader.data
    
    if args.num_samples > 0:
        dataset = dataset[:args.num_samples]
        print(f"Subsampled to {len(dataset)} examples.")

    # Prepare output file
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    
    # Check for existing results to resume
    existing_ids = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl, 'r') as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)['id'])
                except:
                    pass
    print(f"Found {len(existing_ids)} existing results. Resuming...")
    
    remaining_data = [d for d in dataset if d['id'] not in existing_ids]
    
    # Process in batches
    batch_size = args.batch_size
    
    def process_batch(batch_items, current_bs):
        ids = [item['id'] for item in batch_items]
        questions = [item['question'] for item in batch_items]
        img_paths = [item['image_path'] for item in batch_items]
        
        # Load Images
        images = []
        for p in img_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                images.append(Image.new('RGB', (224, 224), color=(0,0,0))) # Safety fallback
        
        grounding_prompts = [model.grounding_prompt] * len(batch_items)
        
        # 1. Full Pass
        y_full_list = model.generate(questions, images, max_new_tokens=args.max_new_tokens)
        
        # 2. Text-Only Pass
        y_text_list = model.generate(questions, None, max_new_tokens=args.max_new_tokens)
        
        # 3. Image-Only Pass (Grounding)
        y_img_list = model.generate(grounding_prompts, images, max_new_tokens=args.max_new_tokens)
        
        results = []
        for idx, item in enumerate(batch_items):
            res = copy.deepcopy(item)
            res.update({
                "y_full": y_full_list[idx]["text"],
                "y_text": y_text_list[idx]["text"],
                "y_img": y_img_list[idx]["text"]
            })
            
            if "meta" not in res:
                res["meta"] = {}
                
            res["meta"].update({
                "model": args.model_id,
                "max_new_tokens": args.max_new_tokens,
                "dtype": "4bit"
            })
            # del res['meta']['meta'] # Removed to fix KeyError
            results.append(res)
        return results

    with open(args.out_jsonl, 'a') as f_out:
        i = 0
        while i < len(remaining_data):
            batch_items = remaining_data[i : i + batch_size]
            
            try:
                batch_results = process_batch(batch_items, batch_size)
                
                for res in batch_results:
                    f_out.write(json.dumps(res) + "\n")
                f_out.flush()
                
                i += batch_size
                print(f"Processed {i}/{len(remaining_data)}", end='\r')
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size > 1:
                    print(f"\nOOM Error! Reducing batch size from {batch_size} to {batch_size // 2}")
                    batch_size = batch_size // 2
                else:
                    print(f"\nOOM even at batch_size=1. Skipping batch starting at {i}.")
                    i += 1 # aggressive skip
            except Exception as e:
                print(f"\nError processing batch: {e}")
                i += batch_size # Skip batch to avoid infinite loop
                
    print("\nInference Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pope_root", required=True, help="Path to POPE dataset JSON")
    parser.add_argument("--images_root", required=True, help="Root dir for images")
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--bf16", action="store_true") # Placeholder, usually handled by wrapper defaults
    
    args = parser.parse_args()
    run_inference(args)
