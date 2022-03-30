import cv2
import numpy as np

img1=cv2.imread(r'C:\Users\LENOVO\Downloads\1st.jpg', cv2.IMREAD_GRAYSCALE)
img2=cv2.imread(r'C:\Users\LENOVO\Downloads\2nd.jpg', cv2.IMREAD_GRAYSCALE)

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



img1_box=list()
img2_box=list()
size=3
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
        ax[i,j].hist(list(img1_box[2*i+j].reshape((size*2)**2)),bins=20, color='green', edgecolor='black')
        ax[i,j].set_title('img1_'+str(2*i+j))
plt.subplots_adjust(hspace=0.3)
plt.show()


fig, ax = plt.subplots(2,2, figsize=(8,6))
for i in range(2):
    for j in range(2):
        ax[i,j].hist(list(img2_box[2*i+j].reshape((size*2)**2)),bins=20, color='#54CAEA', edgecolor='black')
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
        flatten=calc[2*i+j].flatten()
        binX=np.arange(20)*(256//20)
        ax[i,j].plot(binX,flatten,color='r')
        ax[i,j].bar(binX,flatten,width=6,color='b')
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

