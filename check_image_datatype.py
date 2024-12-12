import cv2
import tensorflow as tf
import os

# this file is used to check whether there are any images in the datasets where the file format doesn't work to be
# used in keras. Finally, it removes them.
def check_images(s_dir, ext_list):
    bad_images = []
    bad_ext = []
    s_list = os.listdir(s_dir)
    for klass in s_list:
        klass_path = os.path.join(s_dir, klass)
        print('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list = os.listdir(klass_path)
            for f in file_list:
                f_path = os.path.join(klass_path, f)
                index = f.rfind('.')
                ext = f[index + 1:].lower()
                if ext not in ext_list:
                    print('file ', f_path, ' has an invalid extension ', ext)
                    bad_ext.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img = cv2.imread(f_path)
                        shape = img.shape
                        image_contents = tf.io.read_file(f_path)
                        image = tf.image.decode_jpeg(image_contents, channels=3)
                    except Exception as e:
                        print('file ', f_path, ' is not a valid image file')
                        print(e)
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext


bad_images, bad_ext = check_images("D:/Social", "jpeg")
for file in bad_images:
    os.remove(file)

