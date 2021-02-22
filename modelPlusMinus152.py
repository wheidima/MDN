import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D
from keras.layers import Concatenate
from keras.layers import ZeroPadding3D
from keras import backend as K


def conv3d_bn(x,filters,depth,height,width,padding='SAME',strides=(1,1,1),use_bias=False,use_activation_fn=False,use_bn=True,name=None):
   if name is not None:
      bn_name = name + '_bn'
      conv_name = name + '_conv'
   else:
      bn_name = None
      conv_name = None

   x = Conv3D(filters,(depth,height,width),strides=strides,padding=padding,use_bias=use_bias,name=conv_name)(x)

   if use_bn:
      if K.image_data_format() == 'channels_first':
         bn_axis = 1
      else:
         bn_axis = 4
      x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

   if use_activation_fn:
      x = Activation('relu', name=name)(x)

   return x


def bottleneckPlus(x,filters,depth,stride,name=None):
	
	if K.image_data_format() == 'channels_first':
		bn_axis = 1
	else:
		bn_axis = 4

	#Exploring summarization with length 2
	y1 = MaxPooling3D(pool_size=(depth[0],1,1),strides=(1,1,1),padding='SAME',name=name+'_plus_max_branch_1')(x)
	y1 = Conv3D(filters,(1,3,3),strides=(1,1,1),padding='SAME',use_bias=False,name=name+'_plus_Conv3d_branch_1')(y1)
	y1 = BatchNormalization(axis=bn_axis, scale=False, name=name+'_plus_bn_branch_1')(y1)
	y1 = Activation('relu', name=name+'_plus_relu_branch_1')(y1)

	#Exploring summarization with length 3
	y2 = MaxPooling3D(pool_size=(depth[1],1,1),strides=(1,1,1),padding='SAME',name=name+'_plus_max_branch_2')(x)
	y2 = Conv3D(filters,(1,3,3),strides=(1,1,1),padding='SAME',use_bias=False,name=name+'_plus_Conv3d_branch_2')(y2)
	y2 = BatchNormalization(axis=bn_axis, scale=False, name=name+'_plus_bn_branch_2')(y2)
	y2 = Activation('relu', name=name+'_plus_relu_branch_2')(y2)

	#Exploring summarization with length 4
	y3 = MaxPooling3D(pool_size=(depth[2],1,1),strides=(1,1,1),padding='SAME',name=name+'_plus_max_branch_3')(x)
	y3 = Conv3D(filters,(1,3,3),strides=(1,1,1),padding='SAME',use_bias=False,name=name+'_plus_Conv3d_branch_3')(y3)
	y3 = BatchNormalization(axis=bn_axis, scale=False, name=name+'_plus_bn_branch_3')(y3)
	y3 = Activation('relu', name=name+'_plus_relu_branch_3')(y3)

	y = Concatenate(axis=bn_axis)([y1,y2,y3])

	y = Conv3D(filters,(1,1,1),strides=stride,padding='SAME',use_bias=False,name=name+'_plus_fusion')(y)

	y = BatchNormalization(axis=bn_axis, scale=False, name=name+'_plus_bn_fusion')(y)

	return y

def bottleneckMinus(x,filters,stride,name=None):
	
	if K.image_data_format() == 'channels_first':
		bn_axis = 1
	else:
		bn_axis = 4

	#---Exploring the first difference---
	x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x)
	y1 = Lambda(lambda x: K.abs(x[:,1:,:,:,:]-x[:,:-1,:,:,:]))(x)
	y1 = Conv3D(filters,(1,3,3),strides=(1,1,1),padding='SAME',use_bias=False,name=name+'_minus_Conv3d_branch_1')(y1)
	y1 = BatchNormalization(axis=bn_axis, scale=False, name=name+'_minus_bn_branch_1')(y1)
	y1 = Activation('relu', name=name+'_minus_relu_branch_1')(y1)

	#---Exploring the second difference---
	x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x)
	y2 = Lambda(lambda x: K.abs(x[:,2:,:,:,:]-x[:,:-2,:,:,:]))(x)
	y2 = Conv3D(filters,(1,3,3),strides=(1,1,1),padding='SAME',use_bias=False,name=name+'_minus_Conv3d_branch_2')(y2)
	y2 = BatchNormalization(axis=bn_axis, scale=False, name=name+'_minus_bn_branch_2')(y2)
	y2 = Activation('relu', name=name+'_minus_relu_branch_2')(y2)

	#y = Concatenate([y1,y2],axis=bn_axis)
	y = Concatenate(axis=bn_axis)([y1,y2])
	#y = Concatenate([y1,y2],axis=4)

	y = Conv3D(filters,(1,1,1),strides=stride,padding='SAME',use_bias=False,name=name+'_minus_fusion')(y)

	y = BatchNormalization(axis=bn_axis, scale=False, name=name+'_minus_bn_fusion')(y)
	
	return y

def plusMinus152(input_tensor=None,input_shape=None,classes=4404):

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor,shape=input_shape)
		else:
			img_input = input_tensor

	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 4

	#Downsampling via convolution (spatial only)
	x = Conv3D(64,(7,7,7),strides=(1,2,2),padding='SAME',use_bias=False,name='Conv3d_1_7x7x7')(img_input)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Conv3d_1_7x7x7_bn')(x)
	x = Activation('relu', name='Conv3d_1_7x7x7_relu')(x)

	# Downsampling (spatial only)
	x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='SAME', name='MaxPool2d_1_2x2')(x)

	#-------Layer 2----------
	#Stage 1	
	branch_minus = bottleneckMinus(x,32,stride=(1,1,1),name='Layer2_1')
	branch_plus = bottleneckPlus(x,32,depth=(2,3,4),stride=(1,1,1),name='Layer2_1')
	branch_identity = conv3d_bn(x,32,1,1,1,padding='SAME',use_activation_fn=False,name='Layer2_1')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer2_1_relu_fusion')(x)
	x = Conv3D(32,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer2_1_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer2_1_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer2_1_relu_final')(x)

	#Stage 2	
	branch_minus = bottleneckMinus(x,32,stride=(1,1,1),name='Layer2_2')
	branch_plus = bottleneckPlus(x,32,depth=(2,3,4),stride=(1,1,1),name='Layer2_2')
	branch_identity = conv3d_bn(x,32,1,1,1,padding='SAME',use_activation_fn=False,name='Layer2_2')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer2_2_relu_fusion')(x)
	x = Conv3D(32,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer2_2_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer2_2_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer2_2_relu_final')(x)

	#Stage 3	
	branch_minus = bottleneckMinus(x,32,stride=(1,1,1),name='Layer2_3')
	branch_plus = bottleneckPlus(x,32,depth=(2,3,4),stride=(1,1,1),name='Layer2_3')
	branch_identity = conv3d_bn(x,32,1,1,1,padding='SAME',use_activation_fn=False,name='Layer2_3')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer2_3_relu_fusion')(x)
	x = Conv3D(32,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer2_3_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer2_3_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer2_3_relu_final')(x)

	#-------Layer 3----------
	#Stage 1	
	branch_minus = bottleneckMinus(x,64,stride=(2,2,2),name='Layer3_1')
	branch_plus = bottleneckPlus(x,64,depth=(2,3,4),stride=(2,2,2),name='Layer3_1')
	branch_identity = conv3d_bn(x,64,1,1,1,strides=(2,2,2),padding='SAME',use_activation_fn=False,name='Layer3_1')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_1_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_1_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_1_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_1_relu_final')(x)

	#Stage 2	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_2')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_2')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_2')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_2_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_2_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_2_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_2_relu_final')(x)

	#Stage 3	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_3')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_3')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_3')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_3_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_3_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_3_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_3_relu_final')(x)

	#Stage 4	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_4')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_4')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_4')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_4_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_4_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_4_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_4_relu_final')(x)

	#Stage 5	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_5')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_5')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_5')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_5_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_5_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_5_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_5_relu_final')(x)

	#Stage 6	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_6')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_6')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_6')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_6_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_6_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_6_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_6_relu_final')(x)

	#Stage 7	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_7')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_7')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_7')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_7_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_7_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_7_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_7_relu_final')(x)

	#Stage 8	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_8')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_8')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_8')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_8_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_8_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_8_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_8_relu_final')(x)

	#Stage 9	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_9')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_9')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_9')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_9_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_9_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_9_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_9_relu_final')(x)

	#Stage 10	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_10')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_10')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_10')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_10_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_10_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_10_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_10_relu_final')(x)

	#Stage 11	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_11')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_11')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_11')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_11_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_11_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_11_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_11_relu_final')(x)

	#Stage 12	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_12')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_12')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_12')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_12_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_12_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_12_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_12_relu_final')(x)

	#Stage 13	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_13')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_13')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_13')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_13_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_13_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_13_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_13_relu_final')(x)

	#Stage 14	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_14')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_14')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_14')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_14_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_14_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_14_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_14_relu_final')(x)

	#Stage 15	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_15')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_15')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_15')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_15_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_15_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_15_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_15_relu_final')(x)

	#Stage 16	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_16')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_16')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_16')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_16_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_16_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_16_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_16_relu_final')(x)

	#Stage 17	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_17')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_17')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_17')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_17_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_17_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_17_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_17_relu_final')(x)

	#Stage 18	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_18')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_18')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_18')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_18_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_18_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_18_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_18_relu_final')(x)

	#Stage 19	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_19')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_19')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_19')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_19_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_19_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_19_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_19_relu_final')(x)

	#Stage 20	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_20')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_20')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_20')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_20_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_20_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_20_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_20_relu_final')(x)

	#Stage 21	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_21')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_21')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_21')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_21_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_21_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_21_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_21_relu_final')(x)

	#Stage 22	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_22')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_22')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_22')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_22_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_22_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_22_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_22_relu_final')(x)

	#Stage 23	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_23')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_23')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_23')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_23_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_23_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_23_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_23_relu_final')(x)

	#Stage 24	
	branch_minus = bottleneckMinus(x,64,stride=(1,1,1),name='Layer3_24')
	branch_plus = bottleneckPlus(x,64,depth=(1,2,3),stride=(1,1,1),name='Layer3_24')
	branch_identity = conv3d_bn(x,64,1,1,1,padding='SAME',use_activation_fn=False,name='Layer3_24')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer3_24_relu_fusion')(x)
	x = Conv3D(64,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer3_24_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer3_24_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer3_24_relu_final')(x)

	#-------Layer 4----------
	#Stage 1	
	branch_minus = bottleneckMinus(x,128,stride=(2,2,2),name='Layer4_1')
	branch_plus = bottleneckPlus(x,128,depth=(1,2,3),stride=(2,2,2),name='Layer4_1')
	branch_identity = conv3d_bn(x,128,1,1,1,strides=(2,2,2),padding='SAME',use_activation_fn=False,name='Layer4_1')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_1_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_1_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_1_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_1_relu_final')(x)

	#Stage 2	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_2')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_2')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_2')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_2_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_2_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_2_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_2_relu_final')(x)

	#Stage 3	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_3')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_3')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_3')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_3_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_3_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_3_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_3_relu_final')(x)

	#Stage 4	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_4')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_4')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_4')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_4_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_4_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_4_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_4_relu_final')(x)

	#Stage 5	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_5')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_5')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_5')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_5_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_5_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_5_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_5_relu_final')(x)

	#Stage 6	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_6')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_6')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_6')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_6_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_6_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_6_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_6_relu_final')(x)

	#Stage 7	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_7')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_7')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_7')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_7_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_7_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_7_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_7_relu_final')(x)

	#Stage 8	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_8')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_8')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_8')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_8_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_8_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_8_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_8_relu_final')(x)

	#Stage 9	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_9')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_9')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_9')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_9_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_9_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_9_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_9_relu_final')(x)

	#Stage 10	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_10')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_10')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_10')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_10_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_10_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_10_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_10_relu_final')(x)

	#Stage 11	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_11')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_11')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_11')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_11_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_11_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_11_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_11_relu_final')(x)

	#Stage 12	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_12')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_12')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_12')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_12_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_12_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_12_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_12_relu_final')(x)

	#Stage 13	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_13')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_13')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_13')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_13_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_13_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_13_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_13_relu_final')(x)

	#Stage 14	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_14')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_14')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_14')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_14_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_14_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_14_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_14_relu_final')(x)

	#Stage 15	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_15')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_15')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_15')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_15_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_15_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_15_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_15_relu_final')(x)

	#Stage 16	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_16')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_16')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_16')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_16_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_16_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_16_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_16_relu_final')(x)

	#Stage 17	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_17')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_17')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_17')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_17_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_17_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_17_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_17_relu_final')(x)

	#Stage 18	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_18')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_18')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_18')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_18_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_18_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_18_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_18_relu_final')(x)

	#Stage 19	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_19')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_19')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_19')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_19_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_19_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_19_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_19_relu_final')(x)

	#Stage 20	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_20')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_20')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_20')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_20_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_20_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_20_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_20_relu_final')(x)

	#Stage 21	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_21')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_21')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_21')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_21_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_21_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_21_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_21_relu_final')(x)

	#Stage 22	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_22')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_22')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_22')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_22_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_22_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_22_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_22_relu_final')(x)

	#Stage 23	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_23')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_23')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_23')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_23_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_23_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_23_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_23_relu_final')(x)

	#Stage 24	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_24')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_24')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_24')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_24_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_24_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_24_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_24_relu_final')(x)

	#Stage 25	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_25')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_25')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_25')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_25_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_25_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_25_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_25_relu_final')(x)

	#Stage 26	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_26')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_26')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_26')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_26_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_26_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_26_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_26_relu_final')(x)

	#Stage 27	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_27')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_27')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_27')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_27_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_27_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_27_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_27_relu_final')(x)

	#Stage 28	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_28')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_28')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_28')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_28_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_28_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_28_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_28_relu_final')(x)

	#Stage 29	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_29')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_29')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_29')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_29_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_29_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_29_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_29_relu_final')(x)

	#Stage 30	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_30')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_30')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_30')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_30_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_30_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_30_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_30_relu_final')(x)

	#Stage 31	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_31')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_31')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_31')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_31_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_31_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_31_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_31_relu_final')(x)

	#Stage 32	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_32')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_32')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_32')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_32_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_32_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_32_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_32_relu_final')(x)

	#Stage 33	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_33')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_33')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_33')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_33_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_33_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_33_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_33_relu_final')(x)

	#Stage 34	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_34')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_34')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_34')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_34_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_34_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_34_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_34_relu_final')(x)

	#Stage 35	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_35')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_35')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_35')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_35_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_35_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_35_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_35_relu_final')(x)

	#Stage 36	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_36')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_36')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_36')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_36_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_36_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_36_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_36_relu_final')(x)

	#Stage 37	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_37')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_37')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_37')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_37_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_37_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_37_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_37_relu_final')(x)

	#Stage 38	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_38')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_38')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_38')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_38_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_38_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_38_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_38_relu_final')(x)

	#Stage 39	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_39')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_39')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_39')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_39_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_39_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_39_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_39_relu_final')(x)

	#Stage 40	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_40')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_40')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_40')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_40_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_40_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_40_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_40_relu_final')(x)

	#Stage 41	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_41')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_41')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_41')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_41_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_41_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_41_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_41_relu_final')(x)

	#Stage 42	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_42')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_42')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_42')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_42_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_42_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_42_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_42_relu_final')(x)

	#Stage 43	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_43')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_43')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_43')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_43_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_43_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_43_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_43_relu_final')(x)

	#Stage 44	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_44')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_44')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_44')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_44_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_44_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_44_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_44_relu_final')(x)

	#Stage 45	
	branch_minus = bottleneckMinus(x,128,stride=(1,1,1),name='Layer4_45')
	branch_plus = bottleneckPlus(x,128,depth=(1,1,2),stride=(1,1,1),name='Layer4_45')
	branch_identity = conv3d_bn(x,128,1,1,1,padding='SAME',use_activation_fn=False,name='Layer4_45')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer4_45_relu_fusion')(x)
	x = Conv3D(128,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer4_45_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer4_45_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer4_45_relu_final')(x)

	#-------Layer 5----------
	#Stage 1	
	branch_minus = bottleneckMinus(x,256,stride=(2,2,2),name='Layer5_1')
	branch_plus = bottleneckPlus(x,256,depth=(1,1,2),stride=(2,2,2),name='Layer5_1')
	branch_identity = conv3d_bn(x,256,1,1,1,strides=(2,2,2),padding='SAME',use_activation_fn=False,name='Layer5_1')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer5_1_relu_fusion')(x)
	x = Conv3D(256,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer5_1_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer5_1_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer5_1_relu_final')(x)

	#Stage 2	
	branch_minus = bottleneckMinus(x,256,stride=(1,1,1),name='Layer5_2')
	branch_plus = bottleneckPlus(x,256,depth=(1,1,1),stride=(1,1,1),name='Layer5_2')
	branch_identity = conv3d_bn(x,256,1,1,1,padding='SAME',use_activation_fn=False,name='Layer5_2')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer5_2_relu_fusion')(x)
	x = Conv3D(256,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer5_2_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer5_2_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer5_2_relu_final')(x)

	#Stage 3	
	branch_minus = bottleneckMinus(x,256,stride=(1,1,1),name='Layer5_3')
	branch_plus = bottleneckPlus(x,256,depth=(1,1,1),stride=(1,1,1),name='Layer5_3')
	branch_identity = conv3d_bn(x,256,1,1,1,padding='SAME',use_activation_fn=False,name='Layer5_3')

	x = layers.add([branch_minus,branch_plus])
	x = Activation('relu', name='Layer5_3_relu_fusion')(x)
	x = Conv3D(256,(1,1,1),strides=(1,1,1),padding='SAME',use_bias=False,name='Layer5_3_Conv3d_1x1x1_fusion')(x)
	x = BatchNormalization(axis=channel_axis, scale=False, name='Layer5_3_bn_fusion')(x)

	x = layers.add([x,branch_identity])
	x = Activation('relu', name='Layer5_3_relu_final')(x)

	#----Classification Stage--------
	x = AveragePooling3D((1,4,4),strides=(1,1,1),padding='valid',name='global_avg_pool')(x)

	x = Flatten(name='flatten')(x)

	x = Dense(classes,activation='softmax',name='prediction')(x)

	inputs = img_input
	#Create Model
	model = Model(inputs,x,name='plusMinus152_model')
	return model

