import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10                           #導入kearas範例資料集cifar10
from keras.preprocessing.image import ImageDataGenerator     #導入kearas圖像預處理模組
from matplotlib import pyplot as plt
from lsuv_init import LSUVinit

# 下載并讀取cifar10 資料集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train & x_test:第一筆是資料筆數,第二筆是和第三筆是資料大小,第四筆是RGB三原色,故一般都是3
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) 
# y_train & y_test:第一筆是資料筆數，第二筆是tag數量
# print(x_train[0].shape)     

#轉換資料型態成浮點數
x_train = x_train.astype('float32')      
x_test = x_test.astype('float32')

#資料正規化(將RGB色碼數字最大為255 => 除以255 除以最大數字就是1,所有數字都會介於0~1之間)
x_train = x_train / 255                 
x_test = x_test / 255
y_train


#oneHot 編碼:去掉不重要的特徵值,或將特徵值二元化 ex.00010  01000
NUM_CLASSES = 10  #總過10個分類
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)   
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES) 
y_train[0]

"""### 建立模型"""

model = keras.Sequential()
batch_size = 32
'''
參數：
padding(卷積補值):
“valid”代表只进行有效的卷积，即对边界数据不处理
“same”代表进行补0卷积
“casual”则是代表具有时序关系的卷积(一般不使用)

activation(激活函數、激勵函數):linear、elu、selu、softplus、softsign、relu、tanh、sigmoid、hard_sigmoid(一般常用relu)

advanced_activations(高級激活函數):LeakyReLU、PReLU、ELU、ThresholdedReLU
'''

model.add(keras.layers.Conv2D(32, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(32, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))
model.add(keras.layers.Conv2D(32, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(48, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))
model.add(keras.layers.Conv2D(48, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))   #最大池化
model.add(keras.layers.Dropout(0.25))                 #防止過擬合
model.add(keras.layers.Conv2D(80, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(80, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(80, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(80, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(80, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(128, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(128, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(128, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(128, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.Conv2D(128, padding='same',activation=tf.nn.leaky_relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.GlobalMaxPooling2D())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())       #用於卷積層連接全連接層
model.add(keras.layers.Dense(500, activation=tf.nn.relu))  #普通的全連接層
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))
model.summary() #acc: 0.9479 val_acc: 0.8885

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',             # loss函數選擇
              metrics=['accuracy'])                       #評估標準


# 層序單元方差（LSUV）初始化 - 一種用於深度網絡學習的權重初始化的簡單方法。該方法包括兩個步驟。首先，用正交矩陣預初始化每個卷積或內積層的權重。其次，從第一層到最後一層，將每層輸出的方差歸一化為等於1。 
# 使用不同的激活函數（maxout，ReLU-family，tanh）進行實驗表明，所提出的初始化可以學習非常深的網絡，
# （i）生成測試精度優於或等於標準方法的網絡，
# （ii）至少同樣快作為專門為非常深的網絡提出的複雜方案，如FitNets（Romero等人（2015））和Highway（Srivastava等人（2015））。 
# 在GoogLeNet，CaffeNet，FitNets和Residual網絡上評估性能，並在MNIST，CIFAR-10/100和ImageNet數據集上實現最先進或非常接近的性能。
model = LSUVinit(model,x_train[:batch_size,:,:,:]) 


"""### 圖像擴增 + 訓練模型"""

# 圖像擴增技術
'''
可調參數：
featurewise_center = False，   ＃使输入数据集去中心化（均值为0）
samplewise_center = False，   ＃使输入数据的每个样本均值为0
featurewise_std_normalization = False，   ＃将输入除以数据集的标准差
samplewise_std_normalization = False，   ＃将输入的每个样本除以其自身的标准差
zca_whitening = False，   ＃对输入数据施加ZCA白化
rotation_range = 0，   ＃整数，数据提升时图片随机转动的角度
width_shift_range = 0.1，   ＃浮点数，水平隨機移動圖像（總寬度的某个比例）
height_shift_range = 0.1，   ＃浮点数，垂直隨機移動圖像（總高度的某个比例）
horizo​​ntal_flip = True，   ＃隨機水平翻转
vertical_flip = False，   ＃隨機垂直翻转
shear_range：0.5，   浮点数，剪切强度（逆时针方向的剪切变换角度）
zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
channel_shift_range：浮点数，随机通道偏移的幅度
fill_mode：；‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
preprocessing_function: 将被应用于每个输入的函数。该函数将在图片缩放和数据提升之后运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
data_format：字符串，“channel_first”或“channel_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channel_last”
'''
#設定圖像擴增
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2,vertical_flip=False, horizontal_flip = True)

#將訓練過程寫進tensorbord
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)

datagen.fit(x_train)
#在datagen.flow()生成的批次上安裝模型并開始訓練
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train)/32, epochs=200, 
                             validation_data = (x_test, y_test), workers=8,callbacks=[tbCallBack])

#不擴增圖像直接訓練
# history = model.fit(x_train, y_train, epochs=200)

"""### 測試模型"""

test_loss, test_acc = model.evaluate(x_test, y_test)  #驗證模型

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#image
#img_class = model.predict(image)

"""###   圖形化顯示測試資料"""

label_dict={0:"airplain",1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'hourse',8:'ship',9:'truck'}

for i in range(9):
  plt.subplot(3, 3, i+1)  #顯示的視窗大小 第一個參數為直行,第二個參數為橫列
  plt.imshow(x_train[i])
  plt.title("Class {}".format(y_train[i]))
plt.show()

### 保存模型
model.save('Keras_CFAR10_CNN.h5')   #該檔案保存模型結構、模型權重
#model.save_weights('Keras_CFAR10_CNN_weights.h5')      # 將參數儲存至 HDF5 檔案（不含模型）