import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance


def viewImage(image, name_of_window='img'):
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def download_image(filename_image, image):
    cv.imwrite(filename_image, image)

def delete_small_objects(image, min_size):
    '''
    Returns image, where small objects deleted
    image: binarized
    min_size: int, minimum square to leave on image
    '''
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    new_img = np.zeros((output.shape), dtype=np.ubyte)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            new_img[output == i + 1] = 255
    return new_img

def binarize(img, font, get_contours=False):
    """
    Returns binarized image
    If get_contours=True then also returns contours of objects in binarized image

    font: string
    RED/MONO/MOTLEY
    MONO_ROT - best (use this for mono)
    MY_BLUE/MY_BLUE3
    """
    if font == 'RED':
        _,tresh = cv.threshold(img, 135, 255, cv.THRESH_BINARY)
        med = cv.medianBlur(tresh, 45)
    elif font == 'MONO':
        _,tresh = cv.threshold(img, 74, 255, cv.THRESH_BINARY)
        med = cv.medianBlur(tresh, 45)
    elif font == "MOTLEY":
        _,tresh = cv.threshold(img, 65, 255, cv.THRESH_BINARY)
        med = cv.medianBlur(tresh, 45)
    elif font == "MONO_ROT":
        kernel = np.ones((5,5),np.uint8)
        _,tresh = cv.threshold(img, 74, 255, cv.THRESH_BINARY)
        med = cv.medianBlur(tresh, 45)
        med = cv.morphologyEx(med, cv.MORPH_CLOSE, kernel)
    elif font == "MY_BLUE":
        _,tresh = cv.threshold(img, 137, 255, cv.THRESH_BINARY_INV)
        med = cv.medianBlur(tresh, 45)
        med = cv.medianBlur(med, 45)
    elif font == "MY_BLUE1_3":
        _,tresh = cv.threshold(img, 115, 255, cv.THRESH_BINARY_INV)
        med = cv.medianBlur(tresh, 25)
        med = cv.medianBlur(med, 21)
    elif font == "MY_BLUE3":
        _,tresh = cv.threshold(img, 115, 255, cv.THRESH_BINARY_INV)
        med = cv.medianBlur(tresh, 25)
        med = cv.medianBlur(med, 21)
    elif font=="HSV":
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hs = hsv[:,:,2]
        _, tr = cv.threshold(hs, 92, 255,cv.THRESH_BINARY)
        
        img2 = delete_small_objects(tr, 154990)
        med = cv.medianBlur(img2, 11)
        kernel =  cv.getStructuringElement(cv.MORPH_CROSS,(33,49))
        opening = cv.morphologyEx(med, cv.MORPH_OPEN, kernel)
        
        img3 = delete_small_objects(opening, 145800)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(17,17))
        closing = cv.morphologyEx(img3, cv.MORPH_CLOSE, kernel)
        
        med = cv.medianBlur(closing, 11)
        if get_contours:
            contours, hierarchy = cv.findContours(closing, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
            mask = hierarchy[0,:,3] == -1
            new_contrs = [contour for i, contour in enumerate(contours) if mask[i]]
            return closing, new_contrs
        else:
            return med
    else:
        raise ValueError("only RED, MONO or MOTLEY in font")
    
    
    if get_contours:
        contours, hierarchy = cv.findContours(med, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE) # cv.RETR_EXTERNAL
        return med, contours
    
    return med

def Center_eval(contours):
    """
    Returns centers of puzzle: numpy ndarray
    contours: list of contours of image
    """
    center_list = [np.mean(contr[:,0,:], axis=0) for contr in contours]
    return np.array(center_list)

def Code_print(img, centers, codes=None, detect_object=False):
    """
    Returns image with some text
    
    img: image where to print code
    centers: enumerate object with coordinates, where to put down text
    codes: default=None, then puzzles have serial number
           numpy 2-D array, where 0 column -- peninsula cnt
                                  1 column -- bay cnt
    detect_object: boolean
    if True then serial number is written on puzzles on image
    else codes are written
    """
    out = img.copy()
    for i, center in enumerate(centers):
        if detect_object:
            cv.putText(out, str(i), (int(center[0]), int(center[1])),cv.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 10)
        else:
            text = "P" + str(codes[0,i]) + "B" + str(codes[1,i])
            cv.putText(out, text, (int(center[0])-90, int(center[1])+20),cv.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 10)
    return out

def Code_encoding(pred):
    """
    Returns 2-D numpy array with codes corresponding to it's type in predict vector
    0 column is peninsula cnt
    1 column is bay cnt
    
    pred: numpy 1-D array with types of puzzles
    """
    code = np.ones((pred.shape[0], 2), dtype=int)
    for i, typ in enumerate(pred):
        if typ == 1:
            code[i,0] = 1
            code[i,1] = 3
        elif typ == 2:
            code[i,0] = 1
            code[i,1] = 1
        elif typ == 3:
            code[i,0] = 0
            code[i,1] = 2
        elif typ == 4:
            code[i,0] = 3
            code[i,1] = 0
        elif typ == 5:
            code[i,0] = 1
            code[i,1] = 2
        elif typ == 6 or typ == 9:
            code[i,0] = 2
            code[i,1] = 1
        elif typ == 7:
            code[i,0] = 3
            code[i,1] = 1
        elif typ == 8:
            code[i,0] = 2
            code[i,1] = 2
        elif typ == 1:
            code[i,0] = 1
            code[i,1] = 3
        elif typ == 1:
            code[i,0] = 1
            code[i,1] = 3
    return code.T

def Merge_train(X1, y1, X2, y2):
    '''
    Returns merged matrixs X1 and X2, and arrays y1, y2
    
    X1, X2: matrix, could be different(in that case random indexing will be for wider matrix)
    y1, y2: numpy array
    '''
    if X1.shape[1] < X2.shape[1]:
        min_size = X1.shape[1]
        indeces = np.random.choice(X2.shape[1], min_size, replace=False)
        indeces = np.sort(indeces)
        X2 = X2[:, indeces]
        return np.vstack((X1, X2)), np.hstack((y1,y2))
    else:
        min_size = X2.shape[1]
        indeces = np.random.choice(X1.shape[1], min_size, replace=False)
        indeces = np.sort(indeces)
        X1 = X1[:, indeces]
        return np.vstack((X1, X2)), np.hstack((y1,y2))

def create_X_test(img, font, min_dim_feature):
    '''
    Returns X_test matrix
    
    img: image to test
    font: string, as in binarize("MONO_ROT", "RED"...)
    min_dim_feature: int, to fit 1-dimention to test matrix
    '''
    bin_img, contours = binarize(img, font, True)
    dist_list = [get_distance(contour) for contour in contours]
    indeces = np.random.choice(dist_list[0].shape[0], min_dim_feature, replace=False)
    indeces = np.sort(indeces)
    X_out = dist_list[0][indeces]
    for i in range(1, len(dist_list)):
        indeces = np.random.choice(dist_list[i].shape[0], min_dim_feature, replace=False)
        indeces = np.sort(indeces)
        X_out = np.vstack((X_out, dist_list[i][indeces]))
    
    return X_out

def resize(img, scale):
    '''
    Retuns resized image
    scale: float
    '''
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def rotate(img, angle, scale):
    """
    Returns rotated image in angle
    """
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(img, M, (w, h))

def get_distance(contour):
    """
    Returns numpy array of distances from puzzle's center to it's edge
    contour: numpy array, coordinates of one puzzle
    """
    center = np.mean(contour[:, 0, :], axis=0)
    return np.array([distance.euclidean(vector, center) for vector in contour[:,0,:]])

def concatenate_from_list(list_arrays, min_size=None):
    """
    if min_size=None
    Returns 
    1. concatenated Matrixes(VerticalStack)
    2. output_matrix.Shape[1]
    else 
    Returns only Matrix[:,:min_size], where indexes are random in 1-dim
    """
    if min_size:
        indeces = np.random.choice(list_arrays[0].shape[1], min_size, replace=False)
        indeces = np.sort(indeces)
        X_out = list_arrays[0][:,indeces]
        for i in range(1, len(list_arrays)):
            indeces = np.random.choice(list_arrays[i].shape[1], min_size, replace=False)
            indeces = np.sort(indeces)
            X_out = np.vstack((X_out, list_arrays[i][:, indeces]))
        return X_out

    X_out = list_arrays[0]
    for i in range(1, len(list_arrays)):
        X_out = np.vstack((X_out, list_arrays[i]))
    return X_out, X_out.shape[1]

def roll_dist(dist):
    '''
    Returns matrix of rolled dist-vector
    
    dist: 1-D array of distances from puzzle's center to it's edge
    '''
    dist_list = [np.roll(dist, i * 3) for i in range(dist.size // 3)]
    return concatenate_from_list(dist_list)

def view_code(img, font, detect_object=False, clf=None, X_test=None):
    '''
    Returns image with codes, which model generated
    
    if detect_object=False: returns image with just enumerate puzzles
    else: you must pass classifier and X_test, then codes will be written on every puzzle
    '''
    if detect_object:
        codes = Code_encoding(clf.predict(X_test))
        bin_new, contr = binarize(img, font, True)
        centers_new = Center_eval(contr)
        out_im = Code_print(img, centers_new, codes)
        viewImage(out_im)
        return out_im
    else:
        bin_new, contr = binarize(img, font, True)
        centers_new = Center_eval(contr)
        out_im = Code_print(img, centers_new, detect_object=True)
        viewImage(out_im)
        return out_im

def augmentation_img(img, font, target_list, angle_list):
    '''
    Returns X_Train
    
    font: string(MONO_ROT, RED...)
    target_list: list of numpy arrays of targets types of puzzles
    angle_list: list of 3 numpy arrays, if you want to transform image:
                1st: array of angles
                2nd: array of scale
                3rd: array of indexes in target_list(begins with 1)
    Example:
    if you want to rotate image in 45 and 90, then you have to pass:
    target_list: 3 y_trains with types of puzzles according to their serial number(code puzzles using view_code)
    angle_list = [ np.array([45, 90]), np.array([scale_45, scale_90]), np.array([1, 2])]
    '''
    img_list = [resize(img, scale) for scale in np.arange(1, 1.05, 0.1)]
    contours_list = []
    y_list = [0] * len(img_list)
    
    for i in range(len(img_list)):
        for j, angle in enumerate(angle_list[0]):
            img_list.append(rotate(img_list[i], angle, angle_list[1][j]))
            y_list.append(angle_list[2][j])

    for image in img_list:
        img_bin, contours = binarize(image, font, True)
        #bin_list.append(img_bin)
        contours_list.append(contours)
    
    X_list = []
    label_list = []
    min_dim_feature = 10000

    for i, contours in enumerate(contours_list):
        
        y = target_list[y_list[i]]
        for j, contour in enumerate(contours):
            
            dist = get_distance(contour)
            X, dim_feature = roll_dist(dist)
            
            if dim_feature < min_dim_feature:
                min_dim_feature = dim_feature
            
            X_list.append(X)
            label_list.append(np.array([ y[j] ] * X.shape[0])[:, np.newaxis]) # метка класса умножается на кол-во копий
    
    X = concatenate_from_list(X_list, min_dim_feature)
    label = concatenate_from_list(label_list)[0][:,0]
    
    return X, label, min_dim_feature
