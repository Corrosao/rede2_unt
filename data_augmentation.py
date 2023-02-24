from imagecorruptions import get_corruption_names, corrupt
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import cv2
import shutil
from PIL import Image
from tqdm import tqdm

IMAGE_FOLDER = "/content/rede2_unt/dataset_corrosion_pitting/train/imagens_treinamento/img_original"
MASK_FOLDER = "/content/rede2_unt/dataset_corrosion_pitting/train/imagens_treinamento/img_marcada"
OUTPUT_IMAGE_FOLDER = "/content/rede2_unt/dataset_corrosion_pitting/data_augmentation/train/original"
OUTPUT_MASK_FOLDER = "/content/rede2_unt/dataset_corrosion_pitting/data_augmentation/train/marcada"

print("Iniciando o código e criando as pastas se não existir:")

def sort_image_files(file):
    # Extract the numeric part of the file name
    numeric_part = int(file.split('.')[0])
    return numeric_part

if not os.path.exists(OUTPUT_IMAGE_FOLDER):
    os.makedirs(OUTPUT_IMAGE_FOLDER)

if not os.path.exists(OUTPUT_MASK_FOLDER):
    os.makedirs(OUTPUT_MASK_FOLDER)

print(f"Copiando todas as imagens para a o path:\n{OUTPUT_IMAGE_FOLDER}")
# Copy all images from src folder to dst folder
for image in os.listdir(IMAGE_FOLDER):
    src = os.path.join(IMAGE_FOLDER, image)
    dst = os.path.join(OUTPUT_IMAGE_FOLDER, image)
    shutil.copy2(src, dst)

print(f"Copiando todas as imagens marcadas para a o path:\n{OUTPUT_MASK_FOLDER}")
# Copy all masks from src folder to dst folder
for image in os.listdir(MASK_FOLDER):
    src = os.path.join(MASK_FOLDER, image)
    dst = os.path.join(OUTPUT_MASK_FOLDER, image)
    shutil.copy2(src, dst)

# Get a list of all files in the folder
files = os.listdir(IMAGE_FOLDER)

# Count the number of files in the folder
num_files = len(files)


image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]

sorted_image_files = sorted(image_files, key=sort_image_files)

print("Gerando novas imagens com o imagecorruptions:")
for image_file in tqdm(sorted_image_files):
    image_path = os.path.join(IMAGE_FOLDER, image_file)
    mask_path = os.path.join(MASK_FOLDER, image_file)
    
    image = np.asarray(Image.open(image_path))

    mask = cv2.imread(mask_path)


    for corruption in get_corruption_names():
      if not corruption in ['snow','frost','elastic_transform','pixelate','jpeg_compression','glass_blur']:
        corrupted_image = corrupt(image, corruption_name=corruption, severity=1)

        # add num_files + 1
        num_files += 1
        new_file = f'{num_files}.png'

        imsave(os.path.join(OUTPUT_IMAGE_FOLDER, new_file), corrupted_image)
        # faz um copia da máscara com o mesmo nome da nova imagem
        shutil.copyfile(mask_path, os.path.join(OUTPUT_MASK_FOLDER, new_file))


#---------------------------
import albumentations as A
import cv2
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np

transforms = []
transforms.append(A.Compose([
    A.RandomBrightnessContrast(p=1, brightness_limit=[0.3, 0.3], contrast_limit=0.1)
]))

transforms.append(A.Compose([
    A.Rotate(limit=[60, 60], p=1.0),
]))

# image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]

# sorted_image_files = sorted(image_files, key=sort_image_files)

print("\nGerando novas imagens com o albumentations:")
for transform in transforms:
  for image_file in tqdm(sorted_image_files):
      image_path = os.path.join(IMAGE_FOLDER, image_file)
      mask_path = os.path.join(MASK_FOLDER, image_file)
      
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      mask = cv2.imread(mask_path)

      transformed = transform(image=image, mask=mask)
      transformed_image = transformed['image']
      transformed_mask = transformed['mask']

      alpha = np.ones_like(transformed_mask[:, :, 0], dtype=np.uint8) * 255
      alpha[transformed_mask[:, :, 0] == 255] = 0
      transformed_mask = np.dstack((transformed_mask, alpha))

      # add num_files + 1
      num_files += 1
      new_file = f'{num_files}.png'

      cv2.imwrite(os.path.join(OUTPUT_IMAGE_FOLDER, new_file), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
      cv2.imwrite(os.path.join(OUTPUT_MASK_FOLDER, new_file), transformed_mask)




#----------------------------
# A parte abaixo transforma todas as imagens em preto e branco, pois algumas imagens ficam com uma cor estranha.
# Logo, o código abaixo ajusta isso.
print("Corringo as imagens com cores: ")

# Local onde ficam as imagens a serem corrigidas
folder_path = OUTPUT_IMAGE_FOLDER

# Loop para verificar cada arquivo da pasta
for file_name in tqdm(os.listdir(folder_path)):
    # Trasnforma o caminho todo do arquivo (caminho+arquivo)
    file_path = os.path.join(folder_path, file_name)

    # Processo de transformação da imagem
    img = cv2.imread(file_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Extract the L channel (lightness) from the LAB image
    L = lab[:,:,0]

    # Salva a imagem em preto e branco como um arquivo PNG, reescrevendo por cima do arquivo original
    cv2.imwrite(file_path, L, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(f"\nNúmero total de imagens: {num_files}\n*soma total de imagens copiadas e imagens geradas.")