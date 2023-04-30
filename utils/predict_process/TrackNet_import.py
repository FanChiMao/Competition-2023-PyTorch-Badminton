wrong_1 = "from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img"
revision_1 = "from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img"

with open('../../TrackNetv2/3_in_1_out/predict.py', 'r') as file:
    file_content = file.read()

new_content = file_content.replace(wrong_1, revision_1)


with open('../../TrackNetv2/3_in_1_out/predict.py', 'w') as file:
    file.write(new_content)



