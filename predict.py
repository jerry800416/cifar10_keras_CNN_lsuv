from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10                           #導入kearas範例資料集cifar10
from keras.preprocessing.image import ImageDataGenerator     #導入kearas圖像預處理模組
from matplotlib import pyplot as plt


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

model = load_model('Keras_CFAR10_CNN.h5')   #讀取model

# model.load_weights('Keras_CFAR10_CNN_weights.h5')    # 從 HDF5 檔案載入參數（不含模型）

# model.compile(optimizer=tf.train.AdamOptimizer(),    #編譯model 並設定loss function和評估標準
#               loss='categorical_crossentropy',    
#               metrics=['accuracy'])
# model.summary()

"""### 載入後測試"""

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)