# PYTHON 2.7
from PIL import Image, ImageOps,ImageTk
import Tkinter
import tkMessageBox
import numpy as np
import glob
import os
import face_recognition
from scipy import ndimage
import cv2
#import cv
#import cv2.cv as cv

########################################################################################################
# parameters
model_dir = "./trump_train" # trained model
current = 0
size_x = 256 # for the trained model input
size_y = 256
dilation_size = 1
delay = 100 # for video capture
########################################################################################################

# for video capturing
global last_frame                                      #creating global variable
last_frame = np.zeros((size_x, size_y, 3), dtype=np.uint8)
global cap
cap = cv2.VideoCapture(0)
print("*** LOADED VIDEO CAPTURE MODULE ***")
run_gan = False
f_img = np.zeros((size_x, size_y))
gen_img = np.zeros((size_x, size_y))

# load generator from disk
import tensorflow as tf
sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(model_dir, "model-270000.meta"))
saver.restore(sess, tf.train.latest_checkpoint(model_dir))
b = sess.graph.get_tensor_by_name("batch:1")
op = sess.graph.get_tensor_by_name("convert_outputs/convert_image:0")
feed = {b: None}
print("*** LOADED MODEL FROM DISK ***")

# move between available images
def move(delta, face_ind=0, run_model=True):

    # collect frame
    orig_img = lmain.img#img
    orig_img = np.array(orig_img)
    #print("Shape: ", orig_img.shape)

    # ready for processing with model
    if run_model:
        #################################################################################
        def create_flat_list(list, ind):
            o = []
            if (len(list) == 0):
                print("No face detected.")
                return None
            fields = list[0].keys()
            for f in fields:
                o.append(list[ind][f])

            flat_list = np.asarray([item for sublist in o for item in sublist])

            return flat_list

        # extract facial landmarks
        #orig_img = face_recognition.load_image_file("scaled.png")
        # print(orig_img)
        # import matplotlib.pyplot as plt
        # import matplotlib.image as mpimg
        # #img=mpimg.imread('your_image.png')
        # imgplot = plt.imshow(orig_img)
        # plt.show()
        face_landmarks_list = face_recognition.face_landmarks(orig_img)
        flfl = create_flat_list(face_landmarks_list, face_ind)
        if flfl is None: return np.zeros((size_x, size_y)), np.zeros((size_x, size_y))

        # create binary image
        bw_img = np.zeros(orig_img.shape[0:2])
        bw_img[tuple(np.vstack((flfl[:, 1], flfl[:, 0])))] = 1  # [255, 0, 0]
        bw_img = np.array(ndimage.grey_dilation(bw_img, size=(dilation_size, dilation_size)), dtype=np.uint8)
        bw_img = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2RGB) * 255

        f_img = bw_img#np.ones((256,256, 3), dtype=np.uint8) * 255
        print("Finished face recognition.")
        #################################################################################

        pre = (f_img.astype(np.float32) / 255) * 2 - 1 # ! important or the output will be corrupted
        feed[b] = np.reshape(pre, (1, size_x, size_y, 3))
        r = sess.run(op, feed)
        gen_img = np.reshape(r, (size_x, size_y, 3)).astype(np.uint8)
        print("Finished generation.")
        #################################################################################

    # prepare for visualization in GUI
    last_gen = gen_img
    last_f = f_img
    gen_img = Image.fromarray(gen_img)
    f_img = Image.fromarray(f_img)
    gen_img = ImageTk.PhotoImage(gen_img)
    f_img = ImageTk.PhotoImage(f_img)
    lmain['text'] = "Original"
    label_gen["text"] = "Generated"
    label_gen["image"] = gen_img
    label_gen.photo = gen_img
    label_f["text"] = "Facial Landmarks"
    label_f["image"] = f_img
    label_f.photo = f_img

    return last_gen, last_f



def show_vid(run_model=True):                                        #creating a function
    if not cap.isOpened():                             #checks for the opening of camera
        print("cant open the camera")
    flag, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if flag is None:
        print("Major error!")
    elif flag:
        global last_frame
        last_frame = frame.copy()

    pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    img = img.resize((size_x, size_y), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.img = img
    #lmain.img = face_recognition.load_image_file("test.png")
    lmain.imgtk = imgtk
    if (run_model and run_gan): lmain.run_gan = False
    elif (run_model or run_gan): lmain.run_gan = True
    lmain.configure(image=imgtk)

    # run model inference
    last_gen, last_f = move(0, 0, run_model)

    lmain.after(delay, show_vid)

    return last_gen, last_f

if __name__ == '__main__':
    # root frame
    root = Tkinter.Tk()
    root.title("GAN Live Tool")

    # 1st frame
    im_frame = Tkinter.Frame(root, width=1300, height=300)
    im_frame.pack(side=Tkinter.TOP)

    title_label = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
    title_label.pack(side=Tkinter.TOP)
    title_label["text"] = "Video fed from Camera"
    model_label = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
    model_label.pack(side=Tkinter.TOP)
    model_label["text"] = "Model from: " + model_dir
    lmain = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
    lmain.pack(side=Tkinter.LEFT)
    label_f = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
    label_f.pack(side=Tkinter.LEFT)
    label_gen = Tkinter.Label(im_frame, font=('bold'), compound=Tkinter.TOP)
    label_gen.pack(side=Tkinter.LEFT)

    #run_frame = Tkinter.Frame(root)
    #run_frame.pack(side=Tkinter.BOTTOM)
    # for activating real time transformation
    #Tkinter.Button(run_frame, text="START / STOP", command=lambda: show_vid(True)).pack(side=Tkinter.BOTTOM)

    # just run webcam
    last_gen, last_f = show_vid(True)

    # update
    gen_img = last_gen
    f_img = last_f

    # run
    #move(0)

    root.mainloop()
    cap.release()
