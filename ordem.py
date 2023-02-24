import os
import shutil
from tqdm import tqdm

IMAGEM_ORIGINAL = "/content/images"
MASK_PREENCHIDA = "/content/png"
MASK_BORDA = "/content/png borda"
PREENC_FOLDER = "/content/preenchida"
BORDA_FOLDER = "/content/borda"
ORIGINAL_FOLDER = "/content/original"

# Remove all .xml files from IMAGEM_ORIGINAL folder
for root, dirs, files in os.walk(IMAGEM_ORIGINAL):
    for filename in files:
        if filename.endswith(".xml"):
            os.remove(os.path.join(root, filename))

# Create new directories for preenchida, borda, and original files
os.makedirs(PREENC_FOLDER, exist_ok=True)
os.makedirs(BORDA_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)

# Move preenchida and borda files to the corresponding folders
for preench_file in tqdm(os.listdir(MASK_PREENCHIDA)):
    print(preench_file)
    preench_path = os.path.join(MASK_PREENCHIDA, preench_file)
    if os.path.isfile(preench_path) and preench_file.endswith("_anotada.png"):
        # Remove "_anotada" from the filename
        new_name = preench_file.replace("_anotada.png", ".png")
        borda_file = preench_file.replace("_anotada.png", "_anotada.png")
        borda_path = os.path.join(MASK_BORDA, borda_file)
        print(f"borda_path: {borda_path}")
        print(os.path.exists(borda_path))
        if os.path.exists(borda_path):
            shutil.move(preench_path, os.path.join(PREENC_FOLDER, new_name))
            shutil.move(borda_path, os.path.join(BORDA_FOLDER, borda_file))

            # Find the directory path that contains the file we are looking for
            for root, dirs, files in os.walk(IMAGEM_ORIGINAL):
                if new_name in files:
                    original_dir = root
                    print(original_dir)
                    break
            else:
                # File not found in IMAGEM_ORIGINAL
                continue

            # Move the file to the "original" directory
            original_path = os.path.join(original_dir, new_name)
            shutil.move(original_path, os.path.join(ORIGINAL_FOLDER, new_name))

