import cv2
import numpy as np
import torch
from torchvision import transforms
from alexnet import AlexNet 
import time, sys
from os.path import exists
import os, glob
import matplotlib.pyplot as plt
import matplotlib

# Set to True to save all the individual cells for debugging
# Cells saved in format:
#	<row>_<col>_Lab_<detected_digit>.jpeg
#	Note that <detected_digit> is 0 if no digit is detected
SAVE_CELL = False


'''
Window variables that get updated to show the detected and processed crossword
'''
new_new = cv2.imread("../data/default_images/def1.jpeg")
new_board = cv2.imread("../data/default_images/def1.jpeg")
new_putb = cv2.imread("../data/default_images/def1.jpeg")

'''
solve(), find_empty_cell() and is_valid_move() are used for solving the detected crossowrd
'''
def is_valid_board(board):
    # Check if board is valid
    for i in range(9):
        # check row for duplicate numbers other than zeros
        if len(set(board[i * 9: (i + 1) * 9]) - set([0])) != len([x for x in board[i * 9: (i + 1) * 9] if x != 0]):
            return False
        # check column for duplicate numbers other than zeros
        if len(set(board[i: 81: 9]) - set([0])) != len([x for x in board[i: 81: 9] if x != 0]):
            return False
    # check 3x3 boxes for duplicate numbers other than zeros
    for i in range(3):
        for j in range(3):
            box = [board[row * 9 + col] for row in range(i * 3, i * 3 + 3) for col in range(j * 3, j * 3 + 3)]
            if len(set(box) - set([0])) != len([x for x in box if x != 0]):
                return False
    return True



def solve(board):
	# preliminary check to see if sudoku matrix is wrong
	if not is_valid_board(board):
		print("Invalid board")
		return False
	empty_cell = find_empty_cell(board)
	if not empty_cell:
		return True

	row, col = empty_cell
	for num in range(1, 10):
		if is_valid_move(board, row, col, num):
			board[row * 9 + col] = num
			if solve(board):
				return True
			board[row * 9 + col] = 0
	return False

def find_empty_cell(board):
	for row in range(9):
		for col in range(9):
			if board[row * 9 + col] == 0:
				return (row, col)
	return None

def is_valid_move(board, row, col, num):
	# Check row
	for i in range(9):
		if board[row * 9 + i] == num:
			return False

	# Check column
	for i in range(9):
		if board[i * 9 + col] == num:
			return False

	# Check 3x3 box
	box_row = (row // 3) * 3
	box_col = (col // 3) * 3
	for i in range(3):
		for j in range(3):
			if board[(box_row + i) * 9 + box_col + j] == num:
				return False

	return True

'''
PREPROCESS THE IMAGE TO GET BOARD
- The board is converted to Gray, followed by Guassian blur and adaptive thresholding
- Contours are detected, and the contour resembling a quadrilateral with the largest area is taken to be the crossword itself
'''
def preprocess(board, pts2):
	# Preprocessing the image
	gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			 cv2.THRESH_BINARY_INV, 11, 2)
	
	# Detecting the contours
	try:
		contours,_ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		cnt = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)[0]

		a = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
		maximum = thresh.copy()
		
		cv2.drawContours(maximum, cnt, -1, (255,255,255), 10)
		
		if(len(a)!=4):	
			return pts2, thresh
	except:
		return pts2, thresh
	
	# Rearranging the points
	a = a.reshape(-1, 2)
	c = np.argsort(np.sum(a, axis = 1))
	if(a[c[1]][0] < a[c[2]][0]):
		c[2],c[1] = c[1],c[2]
	c[2], c[3]=c[3], c[2]

	return (np.float32(a[c]), thresh)

'''
GET NUMBERS FROM THE CROPPED BOARD
- Apply adaptive thresholding, and divide the board into 9x9 grid
- Draw a border around cell to remove border artifacts, and pass this to trained AlexNet to recognize the digits
'''
def extract(board_cropped, model, size=28):
	
	transformation = transforms.Compose([
						transforms.Normalize((0.1307,), (0.3081,))
						])
	bw = cv2.cvtColor(board_cropped, cv2.COLOR_BGR2GRAY)
	
	# cv2.imwrite("img.jpg", bw)

	bw = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 9, 17)
	
	cv2.imwrite("img_thresh.jpg", bw)
	cv2.imshow("thresholded", bw)

	bw = bw.reshape(9, size, 9, size).swapaxes(1,2).reshape(81, size, size)
	
	mask = bw[:, size//4 : 3*size//4, size//4 : 3*size//4]
	mask = torch.from_numpy((np.sum(mask != 0, axis=(1,2)) < 10).reshape(9,9))
	
	bw = bw[np.newaxis, :, :, :].swapaxes(0,1)
		
	# Draw a 3px width rectangle around each of the cells
	for i in range(bw.shape[0]):
		cv2.rectangle(bw[i][0], (0, 0), (size-1, size-1), (0, 0, 0), 3)
		
	bw_tensor = torch.from_numpy(bw)
	bw_tensor = bw_tensor.to(torch.float32)
	bw_tensor = transformation(bw_tensor)
	
	out = model(bw_tensor)
	labels = torch.argmax(out, dim=1).reshape(9,9)
	labels[mask] = 0

	if SAVE_CELL:
		if not os.path.exists("./debugging"):
			os.mkdir("./debugging")
		files = glob.glob('./debugging/*')
		for f in files:
			os.remove(f)
		for i in range(9):
			for j in range(9):
				plt.imsave(f"debugging/{i}_{j}_Lab_{labels[i,j]}.jpeg", bw_tensor[9*i+j][0], cmap=matplotlib.cm.gray)

	return labels.numpy()

'''
After solving the sudoku, place numbers on the cropped board to generate the solved sudoku
'''
def putback(numbers, org_nums, board_cropped, size=28):
	global new_new
	# Put the solution back on the board
	new = board_cropped.copy()
	epsilon = size//4
	for i in range(9):
		for j in range(9):
			if numbers[i, j] != 0 and org_nums[i, j] == 0 :
				cv2.putText(new, str(numbers[i, j]), (j*size + 7, i*size + 20),\
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (104, 19, 242), 2)
	# cv2.imshow("new", new)
	new_new = new.copy()
	new_new = cv2.resize(new_new, (540, 540))
	return new

'''
MAIN DRIVER LOOP!
'''
def main():
	global new_board
	global new_putb
	global new_new

	size = 28
	pts2 = np.float32([[0,0],[size*9,0],[size*9,size*9],[0,size*9]])

	model = AlexNet()
	state_dict = torch.load("../data/mnist_alexnet.pt")

	model.load_state_dict(state_dict)
	model.eval()

	LIVE = int(sys.argv[1]) == 1
	frame_no = 0

	"""
	# For live feed
	"""
	if LIVE:
		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
		width  = np.uint16(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
		height = np.uint16(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

	"""
	# For image
	"""
	if not LIVE:
		board = cv2.imread(sys.argv[2])
		height, width, channels = board.shape
		# width = 360
		# height = 540

	prev_numbers = np.zeros((9, 9))
	prev_l = np.zeros((9, 9))
	while True:
		start_time1 = time.time()
		frame_no += 1
		if LIVE:
			ret, board = cap.read()

		# Extract the board from the image
		pts1, thresh = preprocess(board, pts2)
		M = cv2.getPerspectiveTransform(pts1, pts2)
		board_cropped = cv2.warpPerspective(board, M, (size*9,size*9))

		boardFound = False
		if not (pts1==pts2).all():
			boardFound = True
		else:
			new_new =  cv2.imread("../data/default_images/def1.jpeg")
			new_putb = cv2.imread("../data/default_images/def1.jpeg")

		vid = board
		# reshape to 540*360
		new_board = board_cropped.copy()
		new_board = cv2.resize(new_board, (540, 540))

		'''
		Merge all the images together into a single window
		'''
		
		# Get the maximum dimensions
		max_width = max([image.shape[1] for image in [vid, new_new, new_board, new_putb]])
		max_height = max([image.shape[0] for image in [vid, new_new, new_board, new_putb]])

		# Compute the padding
		left_padding_image1 = int((max_width - vid.shape[1])/2)
		right_padding_image1 = max_width - vid.shape[1] - left_padding_image1
		top_padding_image1 = int((max_height - vid.shape[0])/2)
		bottom_padding_image1 = max_height - vid.shape[0] - top_padding_image1

		left_padding_image2 = int((max_width - new_board.shape[1])/2)
		right_padding_image2 = max_width - new_board.shape[1] - left_padding_image2
		top_padding_image2 = int((max_height - new_board.shape[0])/2)
		bottom_padding_image2 = max_height - new_board.shape[0] - top_padding_image2

		left_padding_image3 = int((max_width - new_new.shape[1])/2)
		right_padding_image3 = max_width - new_new.shape[1] - left_padding_image3
		top_padding_image3 = int((max_height - new_new.shape[0])/2)
		bottom_padding_image3 = max_height - new_new.shape[0] - top_padding_image3

		left_padding_image4 = int((max_width - new_putb.shape[1])/2)
		right_padding_image4 = max_width - new_putb.shape[1] - left_padding_image4
		top_padding_image4 = int((max_height - new_putb.shape[0])/2)
		bottom_padding_image4 = max_height - new_putb.shape[0] - top_padding_image4

		# Create a border around each image to match the maximum dimensions and center the images within the border
		border_image1 = cv2.copyMakeBorder(vid, top_padding_image1, bottom_padding_image1, left_padding_image1, right_padding_image1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
		border_image2 = cv2.copyMakeBorder(new_board, top_padding_image2, bottom_padding_image2, left_padding_image2, right_padding_image2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
		border_image3 = cv2.copyMakeBorder(new_new, top_padding_image3, bottom_padding_image3, left_padding_image3, right_padding_image3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
		border_image4 = cv2.copyMakeBorder(new_putb, top_padding_image4, bottom_padding_image4, left_padding_image4, right_padding_image4, cv2.BORDER_CONSTANT, value=(0, 0, 0))


		cv2.putText(border_image1, 'Video', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(border_image2, 'Cropped and Transformed Board', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(border_image3, 'Solved board', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(border_image4, 'Inverse transformed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


		h1 = cv2.hconcat([border_image1, border_image2])
		h2 = cv2.hconcat([border_image3, border_image4])
		combined = cv2.vconcat([h1, h2])
		cv2.namedWindow('Combined Image', cv2.WINDOW_NORMAL)
		cv2.imshow('Combined Image', combined)
		
		
		sameBoard = False

		if boardFound and frame_no % 3 == 0:

			# Extract the numbers from the board
			start_time = time.time()
			numbers = extract(board_cropped, model)

			print("extract time: %s seconds" % (time.time() - start_time))
			print(numbers)
			solvable = True
			
			if (numbers == prev_numbers).all():
				sameBoard = True
				print("Same board!")
			
			start_time=time.time()
			if not sameBoard:
				l = numbers.flatten()
				if solve(l) is False:
					solvable = False
					print("Not solvable!")
			else:
				l = prev_l
			print("solve time: %s seconds" % (time.time() - start_time))

			board1 = board.copy()
			if solvable:
				# Put the solution back on the board
				solved_board = putback(l.reshape((9, 9)), numbers, board_cropped)
				board1 = cv2.warpPerspective(solved_board, M, (width, height), board1, cv2.WARP_INVERSE_MAP, \
						borderMode=cv2.BORDER_TRANSPARENT)
			new_putb = board1

			# save the image
			cv2.imwrite("solved.jpeg", board1)

			print("Time taken: ", time.time() - start_time1, " seconds")

			prev_numbers = numbers
			prev_l = l

		if (cv2.waitKey(2) == ord('q')):
			break
		
	cv2.destroyAllWindows()

if __name__ == '__main__':
	if not exists("../data/mnist_alexnet.pt"):
		print("The Alexnet weights file has not been downloaded. Download the file from the below link and place it in the /data folder.")
	else:
		if len(sys.argv) == 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
			print("Usage:\t`python driver.py <0/1> <path_to_image_if_earlier_was_0>`")
			print("Example:\t`python driver.py 0 ../data/sudoku1.jpeg` for testing the code on a image")
			print("Example:\t`python driver.py 1` for testing the code on live webcam video")
		else:
			main()