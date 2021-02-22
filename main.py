import os
import tensorflow as tf
import numpy as np
import random

from util import wsort,readCSV
from keras.preprocessing import image
import keras
from keras import backend as K
from keras.engine import Model
from keras_vggface import utils
from keras.layers import Flatten,Dense,Input,Lambda,Dropout, Conv3D,Permute

from modelPlusMinus152 import plusMinus152
import psutil
import sys

os.environ['CUDA_VISIBLE_DEVICES']='0'

NUMBER_CLASSES = 8631
DROPOUT_HOLDER = 0.5
WEIGHT_OF_LOSS_WEIGHT = 7e-7
_SAVER_MAX_TO_KEEP = 10
_MOMENTUM = 0.9
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
BATCH = 30
NUM_RGB_CHANNELS = 3
NUM_FRAMES = 16
CHANNELS = 3
_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}


directory = '/DEPRESSION_DATASET/'

dirl = ['cropImages/Training/Freeform/','cropImages/Training/Northwind/','cropImages/Development/Freeform/','cropImages/Development/Northwind/']

dirLabels = ['/DEPRESSION_DATASET/Depression/AVEC2014/AVEC2014_DepressionLabels/AVEC2014_DepressionLabels/Training_DepressionLabels','/DEPRESSION_DATASET/Depression/AVEC2014/AVEC2014_DepressionLabels/AVEC2014_DepressionLabels/Development_DepressionLabels']

dirDevelopment = directory+'cropImages/Development/'
dirDev = ['Freeform/','Northwind/']

dirTesting = directory+'cropImages/Testing/'
dirTest = ['Freeform/','Northwind/']
dirLabelsTest = '/DEPRESSION_DATASET/Depression/AVEC2014/AVEC2014_Labels_Testset/Testing/DepressionLabels/'


def val_generator():
	while True:
		X=np.zeros((BATCH,NUM_FRAMES,112,112,3))
		Y=[]
		
		modality = np.random.randint(2,size=BATCH)
		usuario = np.random.randint(50,size=BATCH)
		for m in range(BATCH):
			users=os.listdir(dirDevelopment+dirDev[modality[m]])
			images=wsort(dirDevelopment+dirDev[modality[m]]+users[usuario[m]]+'/') #all the images
			numImages=len(images)

			imagens = np.random.randint(numImages)
			indice=0
			for j in range(imagens,imagens+64,4):
				imagem = image.load_img(dirDevelopment+dirDevelopment[modality[m]]+users[usuario[m]]+'/'+images[j%numImages],target_size=(112,112))
				imagem=image.img_to_array(imagem)
				# here you put your function to subtract the mean of vggface2 dataset
				imga = utils.preprocess_input(imagem,version=2) #subtract the mean of vggface dataset
				X[m,indice,:,:,:]=imga
				indice=indice+1
			label = readCSV(dirLabels+users[usuario[m]]+'_Depression.csv')
			Y.append(label)
			
		Y=np.array(Y)
		yield X,Y



def generator():
	while True:
		modality = np.random.randint(4,size=BATCH)
		usuario = np.random.randint(50,size=BATCH)

		X = np.zeros((BATCH,NUM_FRAMES,112,112,3))
		Y = []
		Y2=[]
		for i in range(BATCH):
			users = os.listdir(directory+dirl[modality[i]])
				
			images = wsort(directory+dirl[modality[i]]+users[usuario[i]]+'/') #all the images
			numImages = len(images)

			imagens = np.random.randint(numImages)
			indice = 0

			valor = np.random.random()
			if valor < 0.25:
				flagFlip = 1
			elif valor < 0.50 and valor >= 0.25:
				flagFlip = 2
			elif valor < 0.75 and valor >= 0.50:
				flagFlip = 3
			else:
				flagFlip = 4

				
			for j in range(imagens,imagens+64,4):
				imagem=image.load_img(directory+dirl[modality[i]]+users[usuario[i]]+'/'+images[(j)%numImages],target_size=(112,112))
				imagem = image.img_to_array(imagem)
				# here you put your function to subtract the mean of vggface2 dataset
				imga = utils.preprocess_input(imagem,version=2) #subtract the mean of vggface dataset

				if flagFlip == 1:
					X[i,indice,:,:,:] = np.flip(imga,axis=1)
				elif flagFlip == 2:
					X[i,indice,:,:,:] = image.apply_affine_transform(imga,theta=10, channel_axis=2, fill_mode='nearest',cval=0.,order=1)
				elif flagFlip == 3:
					X[i,indice,:,:,:] = np.flip(imga,axis=0)
				else:
					X[i,indice,:,:,:]=imga

				indice = indice+1
			sets = dirl[modality[i]].split('/')[1]
			# You can train the model using Training and Development sets
			if sets == 'Training':
				label = readCSV(dirLabels[0]+'/'+users[usuario[i]]+'_Depression.csv')
			else:
				label = readCSV(dirLabels[1]+'/'+users[usuario[i]]+'_Depression.csv')
			Y.append(label)
			
		
		Y=np.array(Y)
		
		yield X,Y


if __name__ == '__main__':

	sys.setrecursionlimit(2500)
	
	np.random.seed(42)

	rgb_model = plusMinus152(input_shape=(NUM_FRAMES,FRAME_HEIGHT,FRAME_WIDTH,NUM_RGB_CHANNELS),classes=NUMBER_CLASSES)

	last_layer = rgb_model.get_layer('flatten').output
	
	#--FC Layer
	hidden1 = Dense(256,activation='relu',name='hidden1')(last_layer)
	
	#--Regression Layer
	out = Dense(1,activation='linear',name='classifier')(hidden1)

	custom_vgg_model = Model(rgb_model.input,out)

	custom_vgg_model.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=0.001,decay=0.0005))
	custom_vgg_model.fit_generator(generator(),samples_per_epoch=1000,validation_data=val_generator(),validation_steps=10,epochs=1)

	# Change learning rate to 0.0001
	K.set_value(custom_vgg_model.optimizer.learning_rate, 0.0001)
	custom_vgg_model.fit_generator(generator(),samples_per_epoch=1000,validation_data=val_generator(),validation_steps=10,epochs=1)
	
	# Change learning rate to 0.00001
	K.set_value(custom_vgg_model.optimizer.learning_rate, 0.00001)
	custom_vgg_model.fit_generator(generator(),samples_per_epoch=1000,validation_data=val_generator(),validation_steps=10,epochs=1)

	# Verify the prediction for a clip
	users = dirTesting+dirTest[0]+'/203_2/'
	#--Here you read the label for the 'users'
	Y=readCSV(dirLabelsTest+'203_2'+'_Depression.csv')
	buf=0
	numberOfFrames = NUM_FRAMES #
	#img = read the images from the users folder
	img = wsort(users) #all the images
	while (buf < numberOfFrames*4):
		imagem = image.load_img(users+img[buf],target_size=(FRAME_HEIGHT,FRAME_WIDTH))
		imagem = image.img_to_array(imagem)
		#Subtract the mean of VGGFace2 dataset
		#---put your function here
		imga = utils.preprocess_input(imagem,version=2) #here it is the mean value of VGGFace dataset
		X.append(imga)
		buf = buf + 4
	X = np.array(X)
	X = np.expand_dims(X,axis=0)
	prediction = custom_vgg_model.predict(X)
