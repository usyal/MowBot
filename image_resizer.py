from PIL import Image
import os

# Resizing the grass images
for filename in os.listdir("Dataset/Grass"):
    img_location = os.path.join("Dataset/Grass", filename)
    img = Image.open(img_location)
    new_img = img.resize((512, 512), Image.Resampling.LANCZOS)
    new_img_location = os.path.join("Cleaned-Dataset/Grass", f"cleaned-{filename}")
    new_img.save(new_img_location)

# Resizing the non-grass images
for filename in os.listdir("Dataset/Non-Grass"):
    img_location = os.path.join("Dataset/Non-Grass", filename)
    img = Image.open(img_location)
    new_img = img.resize((512, 512), Image.Resampling.LANCZOS)
    new_img_location = os.path.join("Cleaned-Dataset/Non-Grass", f"cleaned-{filename}")
    new_img.save(new_img_location)