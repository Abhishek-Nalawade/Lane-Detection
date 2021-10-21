import cv2
import numpy as np
import matplotlib.pyplot as plt

#out = cv2.VideoWriter('City.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1281,373))

#computes the histogram of the frame along the Y direction i.e. for each column
def histog(img):
    his = list()
    hisx = list()
    for i in range(img.shape[1]):

        coor = np.where(img[:,i] > 0)

        his.append(len(coor[0]))
        hisx.append(i)
        hisarr = np.array(his)

    first_half = his[:100]
    second_half = his[100:]
    max1 = max(first_half)
    ind1 = his.index(max1)
    max2 = max(second_half)
    ind2 = his.index(max2)

    #plt.plot(hisx, his, label = "Original Curve")
    #plt.show()
    return ind1, ind2

#estimates the direction of the car
def direction(xy_left , xy_right):
    #print("in function",xy_left,xy_right)
    min_left = np.amin(xy_left[:,1])
    min_right = np.amin(xy_right[:,1])
    gh = np.where(xy_left == min_left)
    hg = np.where(xy_right == min_right)
    lane_center = (xy_left[gh[0][0],0] + xy_right[hg[0][0],0])/2

    if lane_center > 110:
        return 0
    elif lane_center < 95:
        return 1
    else:
        return 2

#computes the radius of curvature
def radius_of_curvature(coeff_l,coeff_r,xy_left,xy_right):
    radius = ((1+((2*coeff_r[0][0]*xy_right[50][1])+coeff_r[1][0])**2)**(3/2))/abs(2*coeff_r[0][0])
    return radius


#Camera Matrix
K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
#print(K.shape)
#distortion coefficients
D = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

#masking the image based on color values
upper = np.array([255, 255, 255])
lower = np.array([0, 0, 230])

vid = cv2.VideoCapture("road.avi")
while True:
    d, frame = vid.read()
    if frame is None:
        break
    #cv2.imshow("road",frame)

    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
    dst = cv2.undistort(frame, K, D, None, newcameramtx)                     #undistorting the frame
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(dst,(3,3),0)                                #blurring the frame

    crop = blur[:161,:]                                          #removing the sky from the image for further processing
    cropped = blur[160:,:]

    #computes homography and gets the bird eye biew of the road
    pts_src = np.array([[515,50], [260,180], [850,180],[685, 50]])
    pts_dst = np.array([[0, 0],[0, 500],[200, 500],[200, 0]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(cropped, h, (200,500))
    hsv = cv2.cvtColor(im_out, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)                           #applying the mask for white color

    a, b = histog(mask)                                     #computes histogram and returns column number with highest white and pixels
    #cv2.imshow("hom",im_out)


    #cv2.imshow("warped",mask)
    cor_left = np.where(mask[:,a-7:a+8] > 0)                #extracting white points from neighbouring columns
    cor_right = np.where(mask[:,b-7:b+8] > 0)
    x_left = cor_left[0]
    y_left = cor_left[1]
    x_right = cor_right[0]
    y_right = cor_right[1]


    coeff_left = np.polyfit(x_left,y_left,2)                #fitting a polynomial of degree 2 to the points
    coeff_right = np.polyfit(x_right,y_right,2)

    sh_left_coeff = coeff_left.shape
    sh_right_coeff = coeff_right.shape

    #obtaining the Y coordinates from the obtained coefficients from previous step
    x = np.arange(500)
    x = np.reshape(x, (500,1))
    xsq = np.square(x)
    one = np.ones((500,1))
    coeff_left = np.reshape(coeff_left, (sh_left_coeff[0],1))
    coeff_right = np.reshape(coeff_right,(sh_right_coeff[0],1))


    X = np.concatenate((xsq, x, one), axis = 1)
    Y_left = np.dot(X, coeff_left)                      #new Y coordinates to draw the polynomial
    Y_right = np.dot(X, coeff_right)
    Y_left = np.uint8(Y_left)
    Y_right = np.uint8(Y_right)
    Y_left = Y_left + (a-7)
    Y_right = Y_right + (b-7)
    left = np.concatenate((Y_left, x), axis = 1)
    right = np.concatenate((Y_right, x), axis = 1)

    right[:,0] = right[::-1,0]                  #reversing the array to obtain a proper fill of the polynomial
    right[:,1] = right[::-1,1]

    cv2.polylines(im_out, [left], False, (0,255,0), 2)
    cv2.polylines(im_out, [right], False, (0,255,0), 2)
    dup = im_out.copy()

    dir = direction(left, right)                #returns a value which corrsponds to the direction in the list in next line
    lis = ['Right','Left','Straight']
    r = radius_of_curvature(coeff_left,coeff_right, left,right)         #finds the radius of curvature

    points = np.concatenate((left, right), axis = 0)
    cv2.fillPoly(im_out, [points], (0, 0, 0))                   #filling the polynomial with black color so as to obtain a transparent fill of green color

    dif =  dup - im_out                      #getting the difference between the obtained homography image and homography image with polynomial drawn in it
    dif = np.uint8(dif)                     #dif contains only the road now
    #cv2.imshow("tp",dif)
    cort = np.where(dif > 0)
    res12 = dif.copy()
    res12[cort[0][:],cort[1][:],:] = dif[cort[0][:],cort[1][:],:] + np.array([0,20,0])          #adding green values to the road
    res12[res12[:,:,0]>=230] = 255
    res12[res12[:,:,1]>=230] = 255
    res12[res12[:,:,2]>=230] = 255
    #res1 = dup + (2*res)
    res12 = np.uint8(res12)

    #print("shape ",res12.shape)
    res1 = im_out + res12               #adding the road to the homography image with black polynomial
    res1 = np.uint8(res1)               #reulting image with transparent green polynomial
    #cv2.imshow("res12", res12)
    #cv2.imshow("res1",res1)
    #cv2.imshow("poly_homo",im_out)

    #coputes the inverse transform of the road lane back to the original position
    h_inv = np.linalg.inv(h)
    im_out1 = cv2.warpPerspective(res1, h_inv, (1281,212))

    #replacing the pixel values of original frame with new computed inverse transform image
    tem = np.where(im_out1>[10,10,10])
    cropped[tem[0][:], tem[1][:], :] = im_out1[tem[0][:], tem[1][:], :]

    #concatenating the removed image to display the final result
    final = np.concatenate((crop, cropped), axis = 0)

    #printing the values on the image
    final = cv2.putText(final, 'Going {}'.format(lis[dir]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    final = cv2.putText(final, 'Radius of curvature {}'.format(round(r,4)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("inv",final)
    #out.write(final)

    #cv2.waitKey(0)
    cv2.waitKey(1)

#out.release()
vid.release()
cv2.destroyAllWindows()
