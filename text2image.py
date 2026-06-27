import megfile
import pandas as pd
from PIL import Image

def create_image_gallery(images, rows=2, cols=2):
    assert len(images) >= rows * cols, "Not enough images provided!"

    img_height, img_width = images[0].size

    # Create a blank image as the gallery background
    gallery_width = cols * img_width
    gallery_height = rows * img_height
    gallery_image = Image.new("RGB", (gallery_width, gallery_height))

    # Paste each image onto the gallery canvas
    for row in range(rows):
        for col in range(cols):
            img = images[row * cols + col]  # Convert numpy array to PIL image
            x_offset = col * img_width
            y_offset = row * img_height
            gallery_image.paste(img, (x_offset, y_offset))

    return gallery_image

# category to subfolder name
class_item = {
    "Anime_Stylization" : "anime",
    "Portrait" : "human",
    "General_Object" : "object",
    "Text_Rendering" : "text",
    "Knowledge_Reasoning" : "reasoning",
    "Multilingualism" : "multilingualism"
}

image_dir = "images" # 生成的图片保存目录
model_name = "Qwen-Image"
model_path = "/path/to/your/Qwen-Image/model"  # 请替换为实际的模型权重路径

# you can choose the language mode here.
df = pd.read_csv("OneIG-Bench-mini.csv", dtype=str)
# df = pd.read_csv("OneIG-Bench-ZH.csv", dtype=str)

# you can change the grid here. (2, 2), (1, 4), (3, 3) ...
# grid = (1, 2)  # single image
grid = (2, 2)  # standard grid

from inference import inference
for idx, row in df.iterrows():
    
    # you can change the language mode here.
    prompt = row['prompt_en']
    # prompt = row['prompt_cn']

    # you can change the device ids here.
    device_ids = [0, 1, 2, 3]
    # device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    images = []

    for cnt in range(grid[0] * grid[1]):
        image = inference(prompt, model_path, category=row['category'], seed=42 + cnt, device_ids=device_ids)
        # image is suggested to save as PIL format.
        if image is None:
            continue
        images.append(image)

    print(f"Generated {len(images)} images for {row['id']}")

    # If the number of generated images is insufficient, fill the remaining slots with black images.
    total_slots = grid[0] * grid[1]
    if len(images) == 0:
        print(f"Failed to generate any image for {row['id']}, fill with black images of size 1024x1024.")
        # If there are no images at all, fill with black images of size 1024x1024.
        black_img = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
        images.extend([black_img] * total_slots)
    elif len(images) < total_slots:
        print(f"Generated {len(images)} images for {row['id']}, fill with black images using the size of the first image.")
        # If there are some images but not enough, fill with black images using the size of the first image.
        img_w, img_h = images[0].size
        black_img = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
        images.extend([black_img] * (total_slots - len(images)))

    image_gallery = create_image_gallery(images, grid[0], grid[1])
    
    print(f"Created image_gallery for {row['id']}")

    file_path = megfile.smart_path_join(image_dir, class_item[row['category']], model_name, f"{row['id']}.webp")
    with megfile.smart_open(file_path, "wb") as f:
        image_gallery.save(f)  
    
    print(f"Saved image_gallery to {file_path}")