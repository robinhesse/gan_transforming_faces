# PYTHON 2.7
import Image, ImageOps
import ImageTk
import Tkinter
import tkMessageBox
import numpy as np
import glob
import os
import face_recognition
from scipy import ndimage
import cv2
import cv

########################################################################################################
# parameters
img_dir = "./live_tool_images/"
model_dir = "./trump_train"
image_list = sorted(glob.glob(os.path.join(img_dir, "*.*g")))#['results_1.png', 'results_2.png']
current = 0
size = 256, 256
dilation_size = 1
########################################################################################################

# load generator from disk
import tensorflow as tf
sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(model_dir, "model-270000.meta"))
saver.restore(sess, tf.train.latest_checkpoint(model_dir))
b = sess.graph.get_tensor_by_name("batch:1")
op = sess.graph.get_tensor_by_name("convert_outputs/convert_image:0")
feed = {b: None}

# move between available images
def move(delta, face_ind=0, run_model=False):
    global current, image_list
    if not (0 <= current + delta < len(image_list)):
        tkMessageBox.showinfo('End', 'No more image.')
        return
    current += delta
    orig_img = Image.open(image_list[current])
    orig_img = np.array(ImageOps.fit(orig_img, size, Image.ANTIALIAS))

    # ready for processing with model

    if run_model:
        #################################################################################
        print("Running Facial Landmarks for ", image_list[current])
        def create_flat_list(list, ind):
            o = []
            if len(list) == 0:
				print("No face detected.")
            fields = list[0].keys()
            for f in fields:
                o.append(list[ind][f])

            flat_list = np.asarray([item for sublist in o for item in sublist])

            return flat_list

        # extract facial landmarks
        face_landmarks_list = face_recognition.face_landmarks(orig_img)
        flfl = create_flat_list(face_landmarks_list, face_ind)

        # create binary image
        bw_img = np.zeros(orig_img.shape[0:2])
        bw_img[tuple(np.vstack((flfl[:, 1], flfl[:, 0])))] = 1  # [255, 0, 0]
        bw_img = np.array(ndimage.grey_dilation(bw_img, size=(dilation_size, dilation_size)), dtype=np.uint8)
        bw_img = cv2.cvtColor(bw_img, cv.CV_GRAY2RGB) * 255

        f_img = bw_img#np.ones((256,256, 3), dtype=np.uint8) * 255
        print("Finished face recognition.")
        #################################################################################
        print("Running Generator for ", image_list[current])

        pre = (f_img.astype(np.float32) / 255) * 2 - 1 # ! important or the output will be corrupted
        feed[b] = np.reshape(pre, (1, size[0], size[1], 3))
        r = sess.run(op, feed)
        gen_img = np.reshape(r, (size[0], size[1], 3)).astype(np.uint8)
        print("Finished generation.")
        #################################################################################
    else:
        gen_img = np.zeros((size))
        f_img = np.zeros((size))

    # prepare for visualization in GUI
    orig_img = Image.fromarray(orig_img)
    gen_img = Image.fromarray(gen_img)
    f_img = Image.fromarray(f_img)
    orig_img = ImageTk.PhotoImage(orig_img)
    gen_img = ImageTk.PhotoImage(gen_img)
    f_img = ImageTk.PhotoImage(f_img)
    label['text'] = "Original"
    label['image'] = orig_img
    label.photo = orig_img
    label_gen["text"] = "Generated"
    label_gen["image"] = gen_img
    label_gen.photo = gen_img
    label_f["text"] = "Facial Landmarks"
    label_f["image"] = f_img
    label_f.photo = f_img


# root frame
root = Tkinter.Tk()
root.title("GAN Live Tool")

# 1st frame
im_frame = Tkinter.Frame(root, width=1300, height=300)
im_frame.pack(side=Tkinter.TOP)

title_label = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
title_label.pack(side=Tkinter.TOP)
title_label["text"] = "Images from: " + img_dir
model_label = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
model_label.pack(side=Tkinter.TOP)
model_label["text"] = "Model from: " + model_dir
label = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
label.pack(side=Tkinter.LEFT)
label_f = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
label_f.pack(side=Tkinter.LEFT)
label_gen = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
label_gen.pack(side=Tkinter.LEFT)

# 2nd frame
ctrl_frame = Tkinter.Frame(root)
ctrl_frame.pack(side=Tkinter.BOTTOM)

Tkinter.Button(ctrl_frame, text='Previous Picture', command=lambda: move(-1, 0, False)).pack(side=Tkinter.LEFT)
Tkinter.Button(ctrl_frame, text='Next Picture', command=lambda: move(+1, 0, False)).pack(side=Tkinter.LEFT)
Tkinter.Button(ctrl_frame, text='Quit', command=root.quit).pack(side=Tkinter.LEFT)

# 3rd frame
run_frame = Tkinter.Frame(root)
run_frame.pack(side=Tkinter.BOTTOM)

Tkinter.Button(run_frame, text="Create Facial Landmarks and Run Trained Model (Generator)", command=lambda: move(0, 0, True)).pack(side=Tkinter.BOTTOM)

# run
move(0)

root.mainloop()
