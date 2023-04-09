from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

faces_X_train = pathlib.Path('./dataset/train/faces/')
faces_y_train = pathlib.Path('./dataset/train/train_faces.csv')
landmarks_X_train = pathlib.Path('./dataset/train/landmarks')
landmarks_y_train = pathlib.Path('./dataset/train/train_landmarks.csv')

def Solve_Image(img_path, face = True, save_img = False, output_path = os.getcwd()):
    
    '''
    Splits the image into parts and arranges them according to the coordinates given from the dataframe.
    
    : Parameters
    - img_path: Image path. os.path object expected.
    - face: True if face image else False Default True.
    -
    '''
    
    
    if not os.path.exists(img_path):
        print("Plase Provide a valid image path.")
        return
    if face:
        df = pd.read_csv(faces_y_train, dtype = dict((str(i)+str(j), str) for j in range(6) for i in range(6)))
        df.index = df['image']
        df = df.drop(columns = ['image'])
    else:
        df = pd.read_csv(landmarks_y_train, dtype = dict((str(i)+str(j), str) for j in range(6) for i in range(6)))
        df.index = df['image']
        df = df.drop(columns = ['image'])
        
        
    img_name = os.path.split(img_path)[-1]
    
    if not img_name:
        print("Expected an image path.")
        return
    
    if img_name not in df.index:
        print("Expected an Image with labels Present.")
        return 
    
    img = np.asarray(Image.open(img_path))
    new_img = np.zeros_like(img)
    part = img.shape[0]//6
    
    for i in range(6):
        for j in range(6):
            r, c = int(df.loc[img_name, str(i)+str(j)][0]), int(df.loc[img_name, str(i)+str(j)][1])
            new_img[r*part:(r+1)*part, c*part:(c+1)*part] = img[i*part:(i+1)*part, j*part:(j+1)*part]
    if save_img:       
        new_img = Image.fromarray(new_img)
        new_img.save(os.path.join(output_path, img_name))
        print(f"Combining Done! Image saved at {os.path.join(output_path, img_name)}")
    plt.imshow(new_img)
    plt.show()
    
Solve_Image('./dataset/train/faces/024XOWrOBS.jpg', save_img = False)