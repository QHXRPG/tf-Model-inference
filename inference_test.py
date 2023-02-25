import cv2
import tensorflow as tf

'''
converter=tf.lite.TFLiteConverter.from_keras_model_file("keras_model.h5")
tflite_model=converter.convert()
open("tflite_model.tflite", "wb").write(tflite_model)
'''

import os
import cv2
import numpy as np
import time
import tensorflow as tf

test_images = './test_img/'
model_path = "./tflite_model.tflite"

# 模型加载并给tensors分配内存
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print("Input details is: " + str(input_details))
output_details = interpreter.get_output_details()
print("Output details is: " + str(output_details))

file_list = os.listdir(test_images)
# 遍历文件
for file in file_list:
    print('=========================')
    full_path = os.path.join(test_images, file)
    print('full_path:{}'.format(full_path))

    # 只要黑白的，大小控制在(28,28)
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    res_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    # 变成长784的一维数据
    #new_img = res_img.reshape(([1,28,28,1]))
    #image_np_expanded = new_img.astype('float32')

    # 增加一个维度，变为 [1, 784]
    image_np_expanded = np.expand_dims(res_img, axis=0)
    image_np_expanded = np.expand_dims(image_np_expanded, axis=3)
    image_np_expanded = image_np_expanded.astype('float32')  # 类型也要满足要求

    # 填装数据
    model_interpreter_start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], image_np_expanded)

    # 模型推理
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 出来的结果去掉没用的维度
    result = np.squeeze(output_data)
    print('predict result:{}'.format(result))
    # print('result:{}'.format(sess.run(output, feed_dict={newInput_X: image_np_expanded})))

    # 输出结果是长度为10（对应0-9）的一维数据，最大值的下标就是预测的数字
    print('result:{}'.format((np.where(result == np.max(result)))[0][0]))
