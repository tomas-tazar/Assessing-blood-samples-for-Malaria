import cv2, numpy as np
from sklearn.cluster import KMeans

# ================================================================================================================= #

def pre_processing_otsu(im):
    minimum_contour_area = 550
    maximum_contour_area = 10_000
    
    image = cv2.imread(im, cv2.IMREAD_COLOR)  # Read image in color mode
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0) # blur image to reduce noise
    
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # apply Otsu's thresholding
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    regions_of_interest = []
    contours_array = []
    image_for_rois = image.copy()
    
    for contour in contours: # remove contours that are too small or too big
        area = cv2.contourArea(contour)
        if area < maximum_contour_area and area > minimum_contour_area:
            contours_array.append(contour)
        else:
            pass
    
    for contour in contours_array:
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)        
        roi = cv2.bitwise_and(image, mask)
        regions_of_interest.append(roi[y:y+h, x:x+w])        
        
    return regions_of_interest, contours_array, image_for_rois

# ================================================================================================================= #

def pre_processing_kmeans(im, number_of_clusters):
    minimum_contour_area = 550
    maximum_contour_area = 10_000

    image = cv2.imread(im, cv2.IMREAD_COLOR)  # Read image in color mode
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0) # blur image to reduce noise
    lab_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB) # convert image to LAB color space
    l, _, _ = cv2.split(lab_image) # extract L channel
    
    reshaped_image = np.reshape(l, (-1, 1)) # reshape image to 2D array
    kmeans = KMeans(n_clusters=number_of_clusters, n_init="auto", random_state=0).fit(reshaped_image) # apply kmeans clustering
    clustered_image = np.reshape(kmeans.labels_, l.shape) # reshape image to 3D array
    contours, _ = cv2.findContours(clustered_image.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours
    
    regions_of_interest = []
    contours_array = []
    image_for_rois = image.copy()
    
    for contour in contours: # remove contours that are too small or too big
        area = cv2.contourArea(contour)
        if area < maximum_contour_area and area > minimum_contour_area:
            contours_array.append(contour)
        else:
            pass
    
    for contour in contours_array:
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)        
        roi = cv2.bitwise_and(image, mask)
        regions_of_interest.append(roi[y:y+h, x:x+w])        
        
    return regions_of_interest, contours_array, image_for_rois

# ================================================================================================================= #
