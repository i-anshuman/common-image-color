import numpy as np
from sklearn.cluster import KMeans
from lib.plot import show_img_comparison
from lib.palettes import palette_perc

def average_pixel_value(image):
    image_temp = image.copy()
    image_temp[:,:,0], image_temp[:,:,1], image_temp[:,:,2] = np.average(image, axis=(0,1))
    show_img_comparison(image, image_temp)

def highest_pixel_frequency(img):
    img_temp = img.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]
    show_img_comparison(img, img_temp)

def k_mean_cluster(img):
    clt = KMeans(n_clusters=5)
    clt_1 = clt.fit(img.reshape(-1, 3))
    show_img_comparison(img, palette_perc(clt_1))