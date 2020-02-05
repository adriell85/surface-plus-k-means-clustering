import k_means_clustering as km
import cv2

imagem=cv2.imread('4.png')

imagem = km.kmeans_segmentation(imagem, 6)

cv2.imshow('teste',km.kmeans_segmentation(imagem,3))
cv2.waitKey(0)

# cv2.imread('teste',imagem)