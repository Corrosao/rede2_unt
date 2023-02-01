import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.models import load_model


if __name__=="__main__":


  image_directory = '/content/rede2_unt/dataset_corrosion_pitting/test'
  mask_directory = '/content/rede2_unt/dataset_corrosion_pitting/test mascara'
  folder_test_results = '/content/rede2_unt/dataset_corrosion_pitting/test_results'

  image_dataset = []

  path1 = image_directory
  files=sorted(os.listdir(path1))
  for i in tqdm(files):
      imge=cv2.imread(path1+'/'+i,1)   #mudar 0 para 1 em imagens com cor
      print(i)
      imge=np.flip(imge, axis=1)
      image_dataset.append(img_to_array(imge))

  mask_dataset = []
  
  path2 = mask_directory
  files=sorted(os.listdir(path2))
  for j in tqdm(files):
      imge2=cv2.imread(path2+'/'+j,0)   #mudar 0 para 1 em imagens com cor

      imge2=np.flip(imge2, axis=1)

      mask_dataset.append(img_to_array(imge2))
      print(j)

  mask_dataset = np.array(mask_dataset)/255.
  image_dataset = np.array(image_dataset)/255.

  print(f"image_dataset: {len(mask_dataset)}")
  print(f"mask_dataset: {len(mask_dataset)}")

  # from sklearn.model_selection import train_test_split
  #IMPORTANTE  test_size = 0 SIGNIFICA A PORCENTAGEM QUE FICARA COMO TESTE
  # SE  test_size = 0.2, SIGNIFICA QUE 20% DAS IMAGENS SERÃO PARA TESTE E NÃO PARA TREINAMENTO

  # X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.2, random_state = 0)
  # X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 1.0, random_state = 0)


  pre_trained_unet_model = load_model('custom-unetweights-8000epochs_29_09.h5', compile=False)
  my_model = pre_trained_unet_model

 # IMPORTANTE O train_test_split LITERALMENTE ESTÁ SEPARANDO UMA % PARA TESTES
 # O QUE, NESTE CASO AQUI, NÃO É NECESSÁRIO
  X_test = image_dataset
  y_test = mask_dataset

  # for i in tqdm(range(len(X_test))):
  for i, name in enumerate(files, start=0):

    test_img = X_test[i] # Está pegando imagens da pasta treinamento
    ground_truth = y_test[i] # máscara

    test_img_input=np.expand_dims(test_img, 0)
    prediction = (my_model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img, cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')
    #save the file
    if not os.path.exists(folder_test_results):
        os.makedirs(folder_test_results)
    filename = name
    plt.savefig(os.path.join(folder_test_results,filename))

    print(name)
    plt.show()
