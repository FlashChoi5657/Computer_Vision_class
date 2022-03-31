import cv2
import numpy as np

img1=cv2.imread(r'C:\Users\LENOVO\Downloads\1st.jpg')
img2=cv2.imread(r'C:\Users\LENOVO\Downloads\2nd.jpg')

img1=cv2.resize(img1, dsize=(img1.shape[1]//4, img1.shape[0]//4), interpolation=cv2.INTER_AREA)
img2=cv2.resize(img2, dsize=(img2.shape[1]//4, img2.shape[0]//4), interpolation=cv2.INTER_AREA)

click_points=list()

def on_mouse(event, x, y, flags, param):
    global radius

    if event == cv2.EVENT_LBUTTONDOWN:
        print("마우스 좌클릭 좌표:", x, y)
        click_points.append((y,x))
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            radius += 1
        elif radius > 1:
            radius -= 1

radius=3
cv2.imshow('img1',img1)
cv2.setMouseCallback('img1', on_mouse, img1)
cv2.waitKey(0)
cv2.destroyWindow()

cv2.imshow('img2',img2)
cv2.setMouseCallback('img2', on_mouse, img2)
cv2.waitKey(0)
cv2.destroyWindow()

print(len(click_points))

img1_box=list()
img2_box=list()
size=6
for coord in click_points[:4]:
    print(coord)
    y,x = coord
    print(img1[y-size:y+size,x-size:x+size])
    img1_box.append(img1[y-size:y+size,x-size:x+size])


for coord in click_points[4:]:
    print(coord)
    y,x = coord
    print(img2[y-size:y+size,x-size:x+size])
    img2_box.append(img2[y-size:y+size,x-size:x+size])

import matplotlib.pyplot as plt


fig, ax = plt.subplots(2,2, figsize=(8,6))
for i in range(2):
    for j in range(2):
        ax[i,j].hist(list(img1_box[2*i+j].reshape((size*2)**2*3)),bins=20, color='green', edgecolor='black')
        ax[i,j].set_title('img1_'+str(2*i+j))
plt.subplots_adjust(hspace=0.3)
plt.show()


fig, ax = plt.subplots(2,2, figsize=(8,6))
for i in range(2):
    for j in range(2):
        ax[i,j].hist(list(img2_box[2*i+j].reshape((size*2)**2*3)),bins=20, color='#54CAEA', edgecolor='black')
        ax[i,j].set_title('img2_'+str(2*i+j))
plt.subplots_adjust(hspace=0.3)
plt.show()

calc=list()
for coord in click_points[:4]:
    y,x = coord
    patch=img1[y-size:y+size,x-size:x+size]
    cal_hist = cv2.calcHist([patch], channels=[0], mask=None, histSize=[20], ranges=[np.min(patch),np.max(patch)])
    calc.append(cal_hist)
    print(cal_hist)
for coord in click_points[4:]:
    y,x = coord
    patch=img2[y-size:y+size,x-size:x+size]
    cal_hist = cv2.calcHist([patch], channels=[0], mask=None, histSize=[20], ranges=[np.min(patch),np.max(patch)])
    calc.append(cal_hist)
    print(cal_hist)

fig, ax = plt.subplots(2,2, figsize=(8,6))
for i in range(2):
    for j in range(2):
        y,x = click_points[2*i+j]
        patch = img1[y - size:y + size, x - size:x + size]
        # flatten=cal_hist[2*i+j].flatten()
        histColor = ['b', 'g', 'r']
        binX = np.arange(20) * (256 // 20)
        for k in range(3):
            cal_hist = cv2.calcHist([patch], channels=[k], mask=None, histSize=[20],
                                    ranges=[np.min(patch), np.max(patch)])
            ax[i,j].plot(binX,cal_hist,color=histColor[k])
        # ax[i,j].bar(binX,cal_hist,width=6,color='b')
        ax[i,j].set_title('img1_'+str(2*i+j))
plt.subplots_adjust(hspace=0.3)
plt.show()


fig, ax = plt.subplots(2,2, figsize=(8,6))
for i in range(2):
    for j in range(2):
        y,x = click_points[2*i+j+4]
        patch = img2[y - size:y + size, x - size:x + size]
        # flatten=cal_hist[2*i+j].flatten()
        histColor = ['b', 'g', 'r']
        binX = np.arange(20) * (256 // 20)
        for k in range(3):
            cal_hist = cv2.calcHist([patch], channels=[k], mask=None, histSize=[20],
                                    ranges=[np.min(patch), np.max(patch)])
            ax[i,j].plot(binX,cal_hist,color=histColor[k])
        # ax[i,j].bar(binX,cal_hist,width=6,color='b')
        ax[i,j].set_title('img2_'+str(2*i+j))
plt.subplots_adjust(hspace=0.3)
plt.show()


fig, ax = plt.subplots(2,2, figsize=(8,6))
for i in range(2):
    for j in range(2):
        flatten=calc[2*i+j+4].flatten()
        binX=np.arange(20)*(256//20)
        ax[i,j].plot(binX,flatten,color='r')
        ax[i,j].bar(binX,flatten,width=6,color='b')
plt.subplots_adjust(hspace=0.3)
plt.show()


sift = cv2.xfeatures2d.SIFT_create()
kps, des = sift.detectAndCompute(img2, None)
kps_r, des_r = sift.detectAndCompute(img1, None)
kp0 = kps[0]
print("pt=({},{}), size={}, angle={}".format(kp0.pt[0], kp0.pt[1], kp0.size, kp0.angle))

len(kps)



bf = cv2.BFMatcher_create()
matches = bf.knnMatch(des, des_r, k=2)

good = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good.append([m])

np.random.shuffle(good)
image_match = cv2.drawMatchesKnn(
    img2, kps, img1, kps_r, good[:10], flags=2, outImg=img2)

pts_x = [kp.pt[0] for kp in kps]
pts_y = [kp.pt[1] for kp in kps]
pts_s = [kp.size for kp in kps]


plt.imshow(image_match)
plt.title("SIFT 특징점 매칭")
plt.axis("off")
plt.show()