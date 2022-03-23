import cv2
import numpy as np

image= cv2.imread("1.jpg")

#KONTURLARI BULMA
def Contours(image,minArea,filter,img_threshold=[100,100]):

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    canny = cv2.Canny(img_blur, img_threshold[0], img_threshold[0])  # Canny ile kenarları bulmak
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(canny, kernel, iterations=3)   # dial ile genişleme
    img_threshold = cv2.erode(img_dial, kernel,iterations=2)    #erode ile erozyon
    #cv2.imshow('img_threshold',img_threshold)

    #DIŞ KONTURLAR
    contours, hiearchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCont = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True) #köşe noktası bulma
            bbox = cv2.boundingRect(c)
            finalCont.append([len(approx), area, approx, bbox, c])
    finalCont = sorted(finalCont, key = lambda x:x[1], reverse=True) #sırala
    return image, finalCont



def SortPTS(edges):
    # A4 Köşeleri Bulma
    # Noktaları Sıralama
    sorted_pts = np.zeros_like(edges)
    edges = edges.reshape((4, 2))
    sum = edges.sum(1)
    sorted_pts[0] = edges[np.argmin(sum)]
    sorted_pts[3] = edges[np.argmax(sum)]
    diff = np.diff(edges, axis=1)
    sorted_pts[1]= edges[np.argmin(diff)]
    sorted_pts[2] = edges[np.argmax(diff)]

    return sorted_pts

#köşe noktalarıyla yeni çarpık görünütler
#pad ile kenarlara dolgu
def warping (image, points, w, h, pad=20):
    points = SortPTS(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # A4 köşe noktalarını sıralama
    transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_img = cv2.warpPerspective(image, transformation_matrix, (w, h))
    Warp = warped_img[pad:warped_img.shape[0]-pad, pad:warped_img.shape[1]-pad]
    return Warp

image = cv2.resize(image,(0, 0), None, 0.4, 0.4)
cv2.imshow('Original Image', image)


while True:

    #A4 BOYUTLARI
    _contours, conts = Contours(image, minArea=50000, filter=4)

    # Liste boş değilse
    if len(conts) != 0:
        BigOne = conts[0][2]

        Warp = warping(image, BigOne, 711, 987)
        _contours, conts2 = Contours(Warp, minArea=2000, filter=4, img_threshold=[50, 50])


        if len(conts) != 0: #boyut
            for obj in conts2:
                #konturların yükseklik ve genişlik bulmak için dik iki kenarı hesapla
                def Curves(x, y):
                    return np.sqrt(np.square(x[0]-y[0]) + np.square(x[1]-y[1]))

                cv2.polylines(_contours, [obj[2]], True, (255, 0, 255), 3)
                nEdges = SortPTS(obj[2])

                cv2.arrowedLine(_contours, (nEdges[0][0][0], nEdges[0][0][1]), (nEdges[1][0][0], nEdges[1][0][1]),
                                (0, 255, 0), 3, 8, 0, 0.05)
                cv2.arrowedLine(_contours, (nEdges[0][0][0], nEdges[0][0][1]), (nEdges[2][0][0], nEdges[2][0][1]),
                                (0, 255, 0), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(_contours, 'w:{}cm'.format(round((Curves(nEdges[0][0]//3, nEdges[1][0]//3)/10), 1)),
                            (x + 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(_contours, 'h:{}cm'.format(round((Curves(nEdges[0][0]//3, nEdges[2][0]//3)/10), 1)),
                            (x - 110, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        _contours = cv2.resize(_contours, (0, 0), None, 0.7, 0.7)
        cv2.imshow('Size Image', _contours)

    cv2.waitKey(0)