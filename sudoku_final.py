import cv2
import numpy as np
import os
from keras.models import load_model
import pickle
# from solver import Linear_solve as s
from solver2 import solve_sudoku as s

cap = cv2.VideoCapture(-2)
size = 28

# lb = pickle.loads(open('simple_nn_lb.pickle', "rb").read())
pts2 = np.float32([[0,0],[size*9,0],[size*9,size*9],[0,size*9]])

def preprocess(frame):
	gray = cv2.cvtColor(board,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(3,3),0)
	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			 cv2.THRESH_BINARY_INV,3, 3)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5),np.int8),3)

	lines = cv2.HoughLinesP(opening,1,np.pi/180,threshold = 80,minLineLength = 100,maxLineGap = 20)
	threshc = thresh.copy()
	try:
		N = lines.shape[0]
		for i in range(N):
		    x1 = lines[i][0][0]
		    y1 = lines[i][0][1]    
		    x2 = lines[i][0][2]
		    y2 = lines[i][0][3]    
		    threshc = cv2.line(threshc,(x1,y1),(x2,y2),(255,255,255),4)
	except:
		pass
	# cv2.imshow('lines',threshc)
	contours,_ = cv2.findContours(threshc,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	cnt = sorted(contours, key=lambda x: cv2.contourArea(x),reverse = True)[0]

	a = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
	maximum = thresh.copy()
	cv2.drawContours(maximum, cnt, -1, (255,255,255), 10)
	# cv2.imshow('max', maximum)
	if(len(a)!=4):	
		return pts2,thresh
	a = a.reshape(-1, 2)
	c = np.argsort(np.sum(a,axis = 1))
	if(a[c[1]][0] < a[c[2]][0]):
		c[2],c[1] = c[1],c[2]
	c[2],c[3]=c[3],c[2]
	# if(type(thresh) is list): return
	return (np.float32(a[c]),thresh)

def extract(board_cropped,thresh_cropped):
	numbers,x,y = [],size,size
	board = np.zeros((9,9))
	# thresh_cropped = cv2.cvtColor(thresh_cropped,cv2.COLOR_GRAY2BGR)
	model = load_model('model.h5')
	for i in range(9):
		for j in range(9):
			y1, y2, x1, x2 = i*y,(i+1)*y,j*x,(j+1)*x
			# print(y1,y2,x1,x2)
			# cv2.imshow(str(i)+' '+str(j),board_cropped[y1:y2,x1:x2].copy())
			# name = input("name:\n")
			img = board_cropped[y1:y2,x1:x2].copy()#.astype("float") / 255.0
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img.reshape(1,size,size,1)
			# img = img.flatten()
			# img = img.reshape((1, img.shape[0]))
			xyz = thresh_cropped[y1:y2,x1:x2].copy()
			# _, xyz = cv2.threshold(xyz, 70, 255, cv2.THRESH_BINARY)
			# cv2.imshow(str(i)+' '+str(j), xyz)
			if(not (xyz[size//4:3*size//4,size//4:3*size//4]>0).sum()>45):
				# numbers.append(((i,j),0))
				label = "0"
			else:
				xyz = xyz.reshape(1,size,size,1)
				prediction = model.predict_classes(xyz)
				# label = lb.classes_[prediction.argmax(axis=0)[0]]
				label = 1
				# print((i,j),prediction)
				numbers.append(((i,j),int(label)))
				board[i,j] = prediction[0]
			# print(str(i)+"X"+str(j)+" = "+label)
			
	return board

i=0
while(True):
	# board = cv2.resize(cv2.imread('sudoku.jpeg'),(360,540))
	_,board = cap.read()

	pts1,thresh = preprocess(board)
	M = cv2.getPerspectiveTransform(pts1,pts2)
	board_cropped = cv2.warpPerspective(board,M,(size*9,size*9))
	thresh_cropped = cv2.warpPerspective(thresh,M,(size*9,size*9))
	# cv2.imshow("board", board_cropped)
	if(i%3==0):
		b = extract(board_cropped, thresh_cropped)
		print(b)
		sol, index = s(b)
		print(sol,"sol")
	# # if(sol != -1): 

	# # cv2.imshow('thresh_cropped',thresh_cropped)

	# cv2.warpPerspective(board,M,(size*9,size*9),board, cv2.WARP_INVERSE_MAP,cv2.BORDER_TRANSPARENT)
	cv2.imshow('putback',board)
	i+=1
	if (cv2.waitKey(100)==ord('q') ):
		break

cv2.destroyAllWindows()
