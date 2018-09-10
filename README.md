<h1>使用KerasCNN實做CIFAR10(LSUV)準確率:90%</h1><br>
<br>
<br>
參考網站:http://nooverfit.com/wp/%E7%94%A8keras%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E5%87%86%E7%A1%AE%E7%8E%8790%E7%9A%84cifar-10%E9%A2%84%E6%B5%8B%E6%A8%A1%E5%9E%8B/
<br>
<br>
運行環境:Ubuntu16.04LTS<br>
<br>
<br>
<h3>使用方法:</h3><br>
1.請先安裝tensorflow-GPU版,可參考這篇:https://1drv.ms/w/s!AqSCQ8Yo16yIga8Y5kcYJT8Yu_6gZg<br>
2.pip3 install -r requirements.txt<br>
3.訓練：python3 keras_cfar10_cnn(lsuv).py<br>
4.測試：python3 predict.py<br>
<br>
<br>
<h3>基本參數以及紀錄：</h3>
<br>
1.約15層卷積<br>
2.learning rate：0.0001<br>
3.batch size：32<br>
4.epochs：200<br>
4.訓練花費時間：約90分鐘<br>
5.使用設備：gtx1070m<br>
loss: 0.1488 - acc: 0.9479 - val_loss: 0.4511 - val_acc: 0.8885<br>
<br>
<br>
9/10<br>
微調算法<br>
1.最後一層更改為globalmaxpolling<br>
epoch187層時val_acc可以到達90%<br>
loss: 0.1479 - acc: 0.9493 - val_loss: 0.3419 - val_acc: 0.9038<br>
<br>
9/10<br>
微調算法<br>
1.活化函數改為leaky_relu<br>
2.訓練時間延到為2hr左右<br>
epoch190層時val_acc可以到達91%<br>
loss: 0.1448 - acc: 0.9494 - val_loss: 0.3092 - val_acc: 0.9081<br>
