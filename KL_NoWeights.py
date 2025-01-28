import networkx as nx
import matplotlib.pyplot as plt 
import sys
import numpy as np
import math

class Graph:
	
	def __init__(self):
		self.nodes = []
		self.edges = []
	def add_node(self,node):
		self.nodes.append(node)
	def add_edge(self,edge):
		self.edges.append(edge)
		edge.vert[0].adjs.append(edge.vert[1])
		edge.vert[1].adjs.append(edge.vert[0])

	def display(self):
		for e in self.edges:
			print(f'Edge {e.no}: {e.vert[0].name} <--->  {e.vert[1].name}')
	def to_networkx(self):
		G = nx.Graph()
		for n in self.nodes:
			G.add_node(n.name)
		for e in self.edges:
			G.add_edge(e.vert[0].name,e.vert[1].name)
		return G
	def partition(self):
		N = len(self.nodes)
		n1 = int(N/2)
		print(n1)
		self.A = self.nodes[0:n1]
		
		self.B = self.nodes[n1:]
	def swap_partition(self,a1,b1):
		a1_n = None
		b1_n = None
		for i in self.nodes:
			if i.name == a1:
				a1_n = i
			if i.name == b1:
				b1_n = i	
		if a1_n is None or b1_n is None:
			print("Node not found")
			return			
		# Swap the nodes in the partition
		
		for i in range(len(self.A)):
			if self.A[i].name == a1:
				self.A[i] = b1_n
				break
		for i in range(len(self.B)):
			if self.B[i].name == b1:
				self.B[i] = a1_n
				break
		return a1_n,b1_n
		
			

class Edge: 
	no = 0 # Edge Number 
	vert = [] # edge vertices e.g: [A,B]
	def __init__(self,n,v):
		self.no = n
		self.vert = v
	


class Node: 
	name = ""
	adjs = [] #Adjacent Nodes
	swapped = False

	def __init__(self,n):
		self.name = n
		self.adjs = []
		self.swapped = False
	
def get_max_g_val(gval):
	max_val = float('-inf')
	max_idx = -1
	for i in gval:
		if gval[i] > max_val:
			max_val = gval[i]
			max_idx = i
	return max_idx
def check_node_in_partition(node,partition):
	for i in partition:
		if i.name == node:
			return True
	return False
def isEdgeBetweenNodes(node1,node2,edges):
	for i in edges:
		if i.vert[0].name == node1 and i.vert[1].name == node2:
			return True
		if i.vert[0].name == node2 and i.vert[1].name == node1:
			return True
	return False

def print_partition(a,b):
	print("PARTITION A: ")
	for i in a:
		print(i.name)
	print("PARTITION B: ")
	for i in b:
		print(i.name)
	return
def get_cut_size(a,b):
	cut_size = 0
	for i in a:
		for j in i.adjs:
			if j in b:
				cut_size += 1
	return cut_size
def klalgo(g):
	 
	g.partition()
	a = g.A
	b = g.B
	# Calculate the cut size
	cut_size = get_cut_size(a,b)
	cut_cost = cut_size
	print("#######   Iteration 1")
	print(f'Initial Cut Size: {cut_size}')
	# Calculate the gain of each node
	D = {}
	for i in a:
		D[i.name] = 0
	for i in b:
		D[i.name] = 0
	for i in a:
		for j in i.adjs:
			if j in b:
				D[i.name] += 1
			else:
				D[i.name] -= 1
	for i in b:
		for j in i.adjs:
			if j in a:
				D[i.name] += 1
			else:
				D[i.name] -= 1	
	print('D is \n',D)
	gval = {}
	for i in g.A:
		for j in g.B:
			if i.swapped is False and j.swapped is False:
				cxy = 0
				if isEdgeBetweenNodes(i.name,j.name,g.edges):
					cxy = 1
				gval[i.name+'-'+j.name] = D[i.name] + D[j.name] - 2*cxy
	print("GVALS")
	print(gval)
	# Get the edge with the maximum gain
	max_idx = get_max_g_val(gval)
	print(max_idx)
	if max_idx == -1:
		print("No edge to swap")
		return
	node1 = max_idx.split('-')[0]
	node2 = max_idx.split('-')[1]
	
	
	print (f'Swapping Vertices {node1} and {node2}')
	n1,n2 = g.swap_partition(node1,node2)
	cut_size = get_cut_size(g.A,g.B)
	if cut_size >= cut_cost:
		print("Cut size increased. Stopping the algorithm")
		g.swap_partition(node2,node1)
		return
	else:
		cut_cost = cut_size

	if n1 is None or n2 is None:
		print("Node not found")
		return
	n1.swapped = True
	n2.swapped = True
	print_partition(g.A,g.B)
	

	for i1 in range(len(a)-1):
		print(f'######  Iteration {i1+2}')
		# Calculate the cut size
		cut_size = 0
		for i in a:
			for j in i.adjs:
				if j in b:
					cut_size += 1
		
		print(f'Cut Size: {cut_size}')
		D_new = {}
		for i in a:
			if i.swapped is False:
				D_new[i.name] = 0
		for i in b:
			if i.swapped is False:
				D_new[i.name] = 0
		for i in a:
			if i.swapped is False:
				for j in i.adjs:
					if j in b:
						D_new[i.name] += 1
					else:
						D_new[i.name] -= 1
		for i in b:
			if i.swapped is False:
				for j in i.adjs:
					if j in a:
						D_new[i.name] += 1
					else:
						D_new[i.name] -= 1
		gval = {}
		for i in g.A:
			for j in g.B:
				if i.swapped is False and j.swapped is False:
					cxy = 0
					if isEdgeBetweenNodes(i.name,j.name,g.edges):
						cxy = 1
					gval[i.name+'-'+j.name] = D_new[i.name] + D_new[j.name] - 2*cxy
		print("GVALS")
		print(gval)
		# Get the edge with the maximum gain
		max_idx = get_max_g_val(gval)
		if max_idx == -1:
			print("No edge to swap")
			return
		
		node1 = max_idx.split('-')[0]
		node2 = max_idx.split('-')[1]
		

		print (f'Swapping Vertices {node1} and {node2}')

		n1,n2 = g.swap_partition(node1,node2)
		cut_size = get_cut_size(g.A,g.B)
		if cut_size >= cut_cost:
			print("Cut size increased. Stopping the algorithm")
			g.swap_partition(node2,node1)
			return
		else:
			cut_cost = cut_size
		if n1 is None or n2 is None:
			print("Node not found")
			return
		n1.swapped = True
		n2.swapped = True
		print_partition(g.A,g.B)

		# Swap the partitions

def build_graph(nodefile,edgefile):

	graph = Graph()
	if len(sys.argv) <= 2:
		print("Usage: python script.py <nodes file> <edges file>")
		sys.exit(1)
	nodefile = sys.argv[1]
	edgefile = sys.argv[2]

	node_list = []
	with open(nodefile,'r') as nf:
		for i in nf:
			j = i.split(" ")
			j1 = j[0].strip()
			if j1.isalnum() is False:

				print("Node Name Error: Not alphanumeric")
				sys.exit(1)

			node_list.append(j1)

	edge_list = []
	with open(edgefile,'r') as ef:
		for i in ef:
			j = i.strip().split('-')	
			edge_list.append(j)
	
	node_obj_dict = {}
	for n in node_list:
		node_obj_dict[n] = Node(n)

	
	for i in node_obj_dict:
		graph.add_node(node_obj_dict[i])
	edge_obj = []
	for i in range(len(edge_list)):	
		e = edge_list[i]
		n1 = node_obj_dict[e[0]]
		edge_obj.append(Edge(i+1,[node_obj_dict[e[0]],node_obj_dict[e[1]]]))

	for i in edge_obj:
		graph.add_edge(i)
	
	return graph
def draw_bisection(graph,pos):
	a = graph.A
	b = graph.B
	distvals = []
	for i in a:
		for j in b:
			d = math.sqrt((pos[i.name][0]-pos[j.name][0])**2 + (pos[i.name][1]-pos[j.name][1])**2)
			distvals.append([d,i.name,j.name,pos[i.name],pos[j.name]])
	distvals.sort()
	
	pt1 = distvals[0][3]
	pt2 = distvals[0][4]
	pt3 = distvals[1][3]
	pt4 = distvals[1][4]

	
	x1 = (pt1[0] + pt2[0])/2
	y1 = (pt1[1] + pt2[1])/2
	x2 = (pt3[0] + pt4[0])/2
	y2 = (pt3[1] + pt4[1])/2

	

	m = (y2-y1)/(x2-x1)
	#c = y1*x2 - y2*x1
	c = y1 - m * x1

	return m,c

def draw_bisection_line(nx_graph,pos,m,c):

	x_values = [pos[node][0] for node in nx_graph.nodes()]  # Get x coordinates of all nodes
	min_x, max_x = min(x_values), max(x_values)

	# Create the x-range for the line
	x_line = [min_x, max_x]
	y_line = [m * x + c for x in x_line]  # Calculate the corresponding y-values for the line

	# Plot the line on top of the graph
	
	
	plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2)


		


def __main__():
	graph = build_graph(sys.argv[1],sys.argv[2])
	graph.display()
	graph.partition()
	a = graph.A
	b = graph.B

	print("PARTITION A:")
	for i in a:
		print(f'Node {i.name} has adjacents: {len(i.adjs)}')

	print("PARTITION B:")
	for i in b:
		print(f'Node {i.name} has adjacents: {len(i.adjs)}')
	graph_new = build_graph(sys.argv[1],sys.argv[2])
	klalgo(graph_new)
	print("****** FINAL RESULT: ******")
	print("PARTITION A:")
	for i in graph_new.A:
		print(i.name)
	
	print("PARTITION B:")
	for i in graph_new.B:
		print(i.name)

	print(f'FINAL CUT SIZE IS {get_cut_size(graph_new.A,graph_new.B)}')
	print("Drawing the graph")
	nx_graph = graph.to_networkx()
	pos = nx.spring_layout(nx_graph)
	m,c= draw_bisection(graph_new,pos)
	
	nx.draw(nx_graph, pos, with_labels=True, node_size=500, node_color="skyblue")
	draw_bisection_line(nx_graph,pos,m,c)
	plt.show()

if __name__ == "__main__":
    __main__()
