import numpy as np 
from itertools import combinations 
from sympy import *
from sympy.solvers.solveset import linsolve

def solver(l):
	def validate(l):
		y1 = y2 = y3 = np.zeros((9,9),np.int8)
		for i in range(0,9):
			y1[i,l[i]-1] = 1
			y2[i,l[:,i]-1] = 1
			j = i//3
			t = l[3*i:3*i+3,3*j:3*j+3].reshape(-1)
			y3[i,t-1] = 1
		if (y1==0).sum() or (y2==0).sum() or (y3==0).sum():
			return False
		return True

	def filled(l):
		if (l==0).sum():
			return False
		return	True	

	def node_consistency(l,Nodes,x):
		i,j = 3*(x[0]//3),3*(x[1]//3)
		Nodes[x]-= set(l[x[0],:]) 
		Nodes[x]-= set(l[:,x[1]]) 
		Nodes[x]-= set(l[i:i+3,j:j+3].flatten())
		# print(Nodes[x],"After")

		if(Nodes[x]==set()): 
			return False
		return True

	def revise(Nodes,x,y):
		i1, i2, j1, j2 = x[0]//3, y[0]//3, x[1]//3, y[1]//3  
		revised = False		
		if x[0]==y[0] or x[1]==y[1] or (i1==i2 and j1==j2):
			if(Nodes[y]-Nodes[x]!=set()):
				revised = True
			Nodes[x]-=Nodes[y]
		return revised

	def AC3(Nodes):
		queue = []
		for x in Nodes.keys():
			queue+=[(x,t) for t in Neighbors[x]]
		# queue = list(combinations(Nodes.keys(), 2))
		print(queue,"queue")
		while(queue != []):
			(x,y) = queue.pop(0)
			if revise(Nodes, x, y):
				print("Here")
				if Nodes[x] == set():
					return False
				others = Neighbors[x]
				others.discard(y)
				for z in others:
					queue.append((z,x))
		return True	

		print(queue)

	Nodes, Neighbors = {}, {}
	l = np.array(l,int)
	x = np.where(l==0)
	row = np.array(list(x[0]))
	col = np.array(list(x[1]))
	print(row, col)
	y = list(zip(x[0],x[1]))
	numbers = [1,2,3,4,5,6,7,8,9]
	for key in y:
		print(key,type(key))
		Nodes[key] = set(numbers[:])
		n = np.where(np.logical_or(row == key[0],col == key[1],
				np.logical_and(row//3==key[0]//3,col//3==key[1]//3)))

		Neighbors[key] = set(list(zip(row[n],col[n])))
		Neighbors[key].discard(key)
		if(not node_consistency(l, Nodes, key)): return False
	if AC3(Nodes):
		for key in Nodes.keys():
			l[key] = Node[key]

def Linear_solve(l):
	if((l==0).sum()>70): return -1,-1
	# Position = indices -> Variables
	# Variables = indices -> answers
	# Index = Variables -> indices
	x = np.where(l==0)
	pair = list(zip(x[0],x[1]))
	n = len(pair)
	index = np.zeros((9,9),int)-1
	name = np.array([])
	for i in range(len(pair)):
		index[pair[i]] = i
		name = np.append(name,str(i))
	# print(name)
	matrix = np.array([],int)
	rhs = np.array([])
	for i in range(0,9):
		#For rows
		if((l[i,:]==0).sum()):
			equation = np.zeros((1,n+1),int)
			equation[0,index[i,np.where(l[i,:]==0)[0]]] = 1
			equation[0,-1] = 45 - np.sum(l[i,:])
			matrix = np.append(matrix,equation)
			# rhs = np.append(rhs, 45 - np.sum(l[i,:]))
	
	for i in range(0,9):
		# For columns
		if((l[:,i]==0).sum()):
			equation = np.zeros((1,n+1))
			equation[0,index[np.where(l[:,i]==0)[0],i]] = 1
			equation[0,-1] = 45 - np.sum(l[:,i])
			matrix = np.append(matrix,equation)
			# rhs = np.append(rhs, 45 - np.sum(l[:,i]))

	for i in range(0,9):
		# For boxes
		row,col = i%3, i//3
		if((l[3*row:3*row+3,3*col:3*col+3]==0).sum()):
			equation = np.zeros((1,n+1))
			x = np.where(l[3*row:3*row+3,3*col:3*col+3]==0)
			r,c = np.array(x[0])+row*3, np.array(x[1])+col*3
			equation[0,index[r,c]] = 1
			equation[0,-1] = 45 - np.sum(l[3*row:3*row+3,3*col:3*col+3])
			matrix = np.append(matrix,[equation])
	# 		rhs = np.append(rhs, 45 - np.sum(l[3*row:3*row+3,3*col:3*col+3]))
			
	matrix = np.reshape(matrix, (-1,n+1))
	# rhs = rhs.tolist()
	
	mat = Matrix(matrix)
	# print(mat)
	# print(', '.join(x for x in name))
	sym = symbols(', '.join(x for x in name))

	# x, y, z, w = symbols('1, y, z, w')
	# print(sym)
	solution = linsolve(mat, sym)
	solution = np.array(list(solution)[0])
	# print(solution)
	if(len(solution)==0 and n>0):
		return -1,-1
	# for x in solution:
		# if(type(x) is not int): 
		# 	print("type")
		# 	return -1,-1
		# if(x>9): 
		# 	print(">9")
		# 	return -1,-1
	return(solution,index)

board = [
			[5,3,4,6,7,8,9,1,2],
			[6,7,2,1,9,5,3,4,8],
			[1,9,8,3,4,2,5,6,7],
			[8,5,9,7,6,1,4,2,3],
			[4,2,6,8,5,3,7,9,1],
			[7,1,3,9,2,4,8,5,6],
			[9,6,1,5,3,7,2,8,4],
			[2,8,7,4,1,9,6,3,5], # 3
			[0,4,5,2,0,6,1,7,9], # 3,8,9
		]
board = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 0],
	 [6, 0, 0, 1, 9, 5, 0, 0, 0],
	 [0, 9, 8, 0, 0, 0, 0, 6, 0],
	 [8, 0, 0, 0, 6, 0, 0, 0, 3],
	 [6, 0, 0, 8, 0, 3, 0, 0, 1],
	 [7, 0, 0, 0, 2, 0, 0, 0, 6],
	 [0, 6, 0, 0, 0, 0, 2, 8, 0],
	 [0, 0, 0, 4, 1, 9, 0, 0, 5],
	 [0, 0, 0, 0, 8, 0, 0, 7, 9]])

# if(not solver(board)):
# 	print("No Solution")
# else:
# 	for i in board:
# 		print(i)

x,y = Linear_solve(np.array(board))
print(x)
print(y)