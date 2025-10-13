from PIL import Image
import os

valid_extensions = ('.jpg', '.jpeg') # William: Added this because it was adding some .DS_Store files from my anaconda, I will add a .gitignore in the future

# Resizing the grass images
for filename in os.listdir("Dataset/Grass"):
    if filename.lower().endswith(valid_extensions): #William: This currently only checks jpg files
        img_location = os.path.join("Dataset/Grass", filename)
        img = Image.open(img_location)
        new_img = img.resize((224, 224), Image.Resampling.LANCZOS)
        new_img_location = os.path.join("Cleaned-Dataset/Grass", f"cleaned-{filename}")
        new_img.save(new_img_location)

# Resizing the non-grass images
for filename in os.listdir("Dataset/Non-Grass"):
    if filename.lower().endswith(valid_extensions): #William: This currently only checks jpg files
        img_location = os.path.join("Dataset/Non-Grass", filename)
        img = Image.open(img_location)
        new_img = img.resize((224, 224), Image.Resampling.LANCZOS)
        new_img_location = os.path.join("Cleaned-Dataset/Non-Grass", f"cleaned-{filename}")
        new_img.save(new_img_location)