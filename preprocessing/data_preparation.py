import face_recognition
import glob
import numpy as np
import os
from scipy.misc import imsave
from scipy import ndimage
import matplotlib.pyplot as plt
from __future__ import print_function
# %matplotlib

# *** Can we optimize this function? ****
# takes a face_landmarks_list and creates a flat representation
# which can be used for direct visualization
def create_flat_list(list, ind):
    o = []
    fields = list[0].keys()
    for f in fields:
        o.append(list[ind][f])

    flat_list = np.asarray([item for sublist in o for item in sublist])

    return flat_list

# takes a directory containing images and the name_prefix (e.g. "frame")
# and processes all using the face_recognition API
def get_facial_landmarks(selected_face, img_dir, name_prefix):
    # sorting is important, usual glob and sorted won't work as intended
    img_files = sorted(glob.glob(img_dir + "*.*"), key=lambda name: int(name[len(img_dir)+len(name_prefix):-4]))

    img_files = img_files#[0:30]
    
    d = get_identities_for_faces(img_files)

    facial_landmarks = []
    facial_landmarks_flat = []
    for ind, img in enumerate(img_files):
        img = face_recognition.load_image_file(img)
        
        # get the corresponding face (face selection)
        face_loc = face_recognition.face_locations(img)
        a = 1e10
        face_index = -1
        for ind2, face in enumerate(face_loc):
			b = abs(sum(np.asarray(d[selected_face])-np.asarray(face)))
			if b < a: 
				a = b
				face_index = ind2
        
        face_landmarks_list = face_recognition.face_landmarks(img)
        
        # decide based on face selection before        
        face_landmarks_flat_list = create_flat_list(face_landmarks_list, face_index)
        facial_landmarks.append(face_landmarks_list)
        facial_landmarks_flat.append(face_landmarks_flat_list)

        if ind % 10 == 0: print("Image ", ind, "/", len(img_files), " processed.")

    return img_files, facial_landmarks, facial_landmarks_flat

# adds gaussian noise to the facial landmarks
def add_gaussian_noise(fl_flat):
    noisy_fl_flat = []
    for fl in fl_flat:
        # two dimensional gaussian with mean at 0,0 and covariance
        s = np.random.multivariate_normal((0, 0), [[1,0], [0,1]], (len(fl)))
        noisy_fl_flat.append(fl + s)
    return noisy_fl_flat

# creates a subplot containing the original image
# the image with recognized facial landmarks and solely the facial landmarks
# and the nosiy edition of the latter
def show_processed_images(size, ind, imgs, fl_flat, noisy_fl_flat):

    s = size
    img = face_recognition.load_image_file(imgs[ind])
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(img)
    plt.scatter(fl_flat[ind][:,0], fl_flat[ind][:,1], s=s)
    # empty background
    a = np.zeros(img.shape[0:2])
    plt.subplot(223)
    plt.imshow(a)
    plt.scatter(fl_flat[ind][:,0], fl_flat[ind][:,1], s=s)
    plt.subplot(224)
    plt.imshow(a)
    plt.scatter(noisy_fl_flat[ind][:,0], noisy_fl_flat[ind][:,1], s=s)
    plt.show()

# saves the different variations for a given frame properly
def save_variations(img_dir, dilation_size, ind, imgs, fl_flat, nosiy_fl_flat):

    img = face_recognition.load_image_file(imgs[ind])
    fn = imgs[ind].split(".")
    
    # for getting them in directory for pix2pix
    # make sure the folders specified below exist
    fn = os.path.basename(imgs[ind])
    
    # 1st variation: img with dots
    f1 = img_dir + "A/" + fn #fn[0] + "_fl." + fn[1]
    img[tuple(np.vstack((fl_flat[ind][:, 1], fl_flat[ind][:, 0])))] = [255, 0, 0]
    img[:, :, 0] = ndimage.grey_dilation(img[:, :, 0], size=(dilation_size, dilation_size)) # pretty creative and hacky: dilation over the red dots channel
    imsave(f1, img)
    # 2nd variation: black with dots
    f2 = img_dir + "B1/" + fn #fn[0] + "_b." + fn[1]
    a = np.zeros(img.shape[0:2])
    a[tuple(np.vstack((fl_flat[ind][:, 1], fl_flat[ind][:, 0])))] = 1#[255, 0, 0]
    a = ndimage.grey_dilation(a, size=(dilation_size,dilation_size))
    imsave(f2, a)
    # 3rd variation: black with noisy dots
    f3 = img_dir + "B2/" + fn #fn[0] + "_bn." + fn[1]
    a = np.zeros(img.shape[0:2])
    a[tuple(np.vstack((nosiy_fl_flat[ind][:, 1].astype(int), nosiy_fl_flat[ind][:, 0].astype(int))))] = 1#[255, 0, 0]
    a = ndimage.grey_dilation(a, size=(dilation_size,dilation_size))
    imsave(f3, a)

# create the concatenated image pairs which can be used
# for training the pix2pix network
def create_pairs(ind, imgs):
    # load images
    fn = imgs[ind].split(".")
    f0 = imgs[ind]
    f1 = fn[0] + "_b." + fn[1]
    f2 = fn[0] + "_bn." + fn[1]
    i0 = face_recognition.load_image_file(f0)
    i1 = face_recognition.load_image_file(f1)
    i2 = face_recognition.load_image_file(f2)
    # create pairs
    p1 = np.hstack((i0, i1))
    p2 = np.hstack((i0, i2))
    imsave(f1.split(".")[0] + "_pair." + fn[1], p1)
    imsave(f2.split(".")[0] + "_pair." + fn[1], p2)

# create identities for faces from first frame
def get_identities_for_faces(imgs):
	# get first frame
	img = face_recognition.load_image_file(imgs[0])
	# get face locations
	face_loc = face_recognition.face_locations(img)
	# create dict
	d = dict(zip(range(len(face_loc)), face_loc))
	return d
	

# process images
img_dir = '/home/robinhesse/Schreibtisch/image_processing/data/videos/trump_imgs/'#'data/videos/trump_imgs/'
imgs, fl, fl_flat = get_facial_landmarks(0, img_dir, "frame")
noisy_fl_flat = add_gaussian_noise(fl_flat)

# don't allow out of bounds
noisy_fl_flat = np.ceil(noisy_fl_flat)
flats = [fl_flat, noisy_fl_flat]
for flat in flats:
	for a in flat:
		for b in a:
			if b[0] >= 1280: b[0] = 1280-1
			elif b[0] < 0: b[0] = 0
			if b[1] >= 720: b[0] = 720-1
			elif b[1] < 0: b[1] = 0
	
# save all
for i in range(len(imgs)):
	if i % 10 == 0: print("processed ", i, "/", len(imgs))
	save_variations(img_dir, 3, i, imgs, fl_flat, noisy_fl_flat)
