import torch
from modelscope import DiffusionPipeline
from PIL import Image

CATEGORY_SIZE_MAP = {
    "Anime_Stylization": (1328, 1328),
    "Portrait": (1152, 1472),
    "General_Object": (1328, 1328),
    "Text_Rendering": (1664, 928),
    "Knowledge_Reasoning": (1664, 928),
    "Multilingualism": (1328, 1328)
}

SYSTEM_PROMPT = """
You are an expert image generator. Your task is to create high-quality images based on the user's text description.
"""

_pipe_cache = {}

def _get_pipe(model_path, device_ids):
    cache_key = (model_path, tuple(device_ids))
    if cache_key not in _pipe_cache:
        torch_dtype = torch.bfloat16
        
        # 设置可见 GPU
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
        
        if len(device_ids) > 1:
            device_map = "balanced"
            print(f"Using model parallel across GPUs: {device_ids}")
        else:
            device_map = {"": f"cuda:0"}  # 因为已经设置了 CUDA_VISIBLE_DEVICES，所以用 cuda:0
            print(f"Using single GPU: cuda:{device_ids[0]}")
        
        pipe = DiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        print(f"Model loaded from: {model_path}")
        _pipe_cache[cache_key] = pipe
    
    return _pipe_cache[cache_key]

def inference(
    prompt, 
    model_path, 
    category=None, 
    seed=None, 
    device_ids=None,
    use_system_prompt=True
):
    if device_ids is None:
        device_ids = [0]
    
    if category is not None:
        width, height = CATEGORY_SIZE_MAP.get(category, (1328, 1328))
    else:
        width, height = 1328, 1328
    
    if use_system_prompt:
        full_prompt = f"{SYSTEM_PROMPT.strip()}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    pipe = _get_pipe(model_path, device_ids)
    
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None
    
    try:
        image = pipe(
            prompt=full_prompt,
            negative_prompt=" ",
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=generator
        ).images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    test_cases = [
        ("Anime_Stylization", "anime girl with long hair, blue eyes, smiling"),
        ("Text_Rendering", "A bar chart showing sales data for 2024"),
        ("Portrait", "A professional portrait of a business person")
    ]
    
    device_ids = [0, 1, 2, 3]
    
    for category, prompt in test_cases:
        image = inference(prompt, model_path, category=category, seed=42, device_ids=device_ids)
        if image:
            image.save(f"test_{category}.png")
            print(f"Saved test_{category}.png (size: {image.size})")
        else:
            print(f"Failed to generate image for {category}")