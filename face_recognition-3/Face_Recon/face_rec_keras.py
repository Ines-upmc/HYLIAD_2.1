import bz2
import os

from urllib.request import urlopen


def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)

dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)


from model import create_model

nn4_small2 = create_model()

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer

# Input for anchor, positive and negative images
in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

# Output for anchor, positive and negative embedding vectors
# The nn4_small model instance is shared (Siamese network)
emb_a = nn4_small2(in_a)
emb_p = nn4_small2(in_p)
emb_n = nn4_small2(in_n)

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a-p), axis=-1)
        n_dist = K.sum(K.square(a-n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

# Layer that computes the triplet loss from anchor, positive and negative embedding vectors
triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])

# Model that can be trained with anchor, positive negative images
nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)

from data import triplet_generator
import tensorflow as tf

# triplet_generator() creates a generator that continuously returns 
# ([a_batch, p_batch, n_batch], None) tuples where a_batch, p_batch 
# and n_batch are batches of anchor, positive and negative RGB images 
# each having a shape of (batch_size, 96, 96, 3).
generator = triplet_generator() 

### Training the model from scratch
#nn4_small2_train.compile(loss=None, optimizer='adam')
#nn4_small2_train.fit_generator(generator, epochs=10, steps_per_epoch=100)


### Using a pretrained model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')


## Custom Dataset

import numpy as np
import os.path

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

#metadata = load_metadata('images')
metadata = load_metadata('/home/ubuntu/3-HYLIAD_2.1/face_recognition-3/images/keras')
#metadata = load_metadata('/home/ubuntu/3-HYLIAD_2.1/face_recognition-3/images/tmp_lfw')

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib
import dlib


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

K = 28
#jc_orig = load_image(metadata[77].image_path())
jc_orig = load_image(metadata[K].image_path())
#jc_orig = dlib.resize_image(jc_orig, 250, 250)

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
def graph1():
    plt.subplot(131)
    plt.imshow(jc_orig)

    # Show original image with bounding box
    plt.subplot(132)
    plt.imshow(jc_orig)
    plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

    # Show aligned image
    plt.subplot(133)
    plt.imshow(jc_aligned);
    plt.show()
    return

#graph1()

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    print (i)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    #plt.suptitle("Distance = "+distance(embedded[idx1], embedded[idx2]))
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));    
    plt.show()
    return

#show_pair(15, 16)
#show_pair(15, -12)


#graph1()
#show_pair(77, 78)
#show_pair(77, 50)

from sklearn.metrics import f1_score, accuracy_score

distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise

num = len(metadata)

for i in range(num - 1):
    for j in range(i + 1, num):
        distances.append(distance(embedded[i], embedded[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)
        
distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

def graph2():
    # Plot F1 score and accuracy as function of distance threshold
    plt.plot(thresholds, f1_scores, label='F1 score');
    plt.plot(thresholds, acc_scores, label='Accuracy');
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
    #plt.title("Accuracy at threshold "+opt_tau+" = "+opt_acc);
    plt.xlabel('Distance threshold')
    plt.legend();
    #plt.show()
    
    dist_pos = distances[identical == 1]
    dist_neg = distances[identical == 0]

    plt.figure(figsize=(12,4))
    
    plt.subplot(121)
    plt.hist(dist_pos)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (pos. pairs)')
    plt.legend();

    plt.subplot(122)
    plt.hist(dist_neg)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (neg. pairs)')
    plt.legend();
    plt.show()
    return

#Face recognition

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

targets = np.array([m.name for m in metadata])

encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
X_train = embedded[train_idx]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
svc = LinearSVC()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))

print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
#print("KNN accuracy = "+acc_knn+", SVM accuracy = "+acc_svc)



def stats():
    import numpy as np
    from collections import Counter
    imgs = [load_image(metadata[i].image_path()) for i in range(len(metadata))]
    imgs = np.array(imgs)
    imgs_n = ['_'.join(str(metadata[i]).split('/')[-1].split('_')[0:2]) for i in range(len(metadata))]
    imgs_n = np.array(list(Counter(imgs_n)))
    from sklearn.metrics import classification_report
    #print(len(np.ravel(X_train)),len(np.ravel(X_test)),len(np.ravel(y_train)),len(np.ravel(y_test)))
    print(classification_report(np.ravel(y_train), np.ravel(y_test), target_names=imgs_n))
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    mat = confusion_matrix(np.ravel(y_train), np.ravel(y_test))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=imgs_n,
                yticklabels=imgs_n)
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    plt.show()
    return


import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

def graph3():
    example_idx = 13

    #example_image = load_image(metadata[test_idx][example_idx].image_path())
    example_image =  cv2.imread("/home/ubuntu/3-HYLIAD_2.1/face_recognition-3/images/tmp_images/INES_KHELIFI_28042022-15h.jpg")
    example_prediction = svc.predict([embedded[test_idx][example_idx]])
    example_identity = encoder.inverse_transform(example_prediction)[0]

    plt.imshow(example_image)
    plt.title(f'Recognized as {example_identity}');
    #plt.title("Recognized as "+example_identity)
    plt.show()
    return

##Data Visualisation

from sklearn.manifold import TSNE

def graph4():
    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t.split('_')[0])   

    #plt.legend(bbox_to_anchor=(1, 1));
    #lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('capture334.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    return

from mpl_toolkits import mplot3d
from sklearn.svm import SVC # "Support vector classifier"


graph1()
show_pair(K, K+1)
show_pair(K, 119)
show_pair(K, 45)
graph2()
#graph3()
graph4()
stats()
