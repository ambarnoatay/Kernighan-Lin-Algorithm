import networkx as nx
import matplotlib.pyplot as plt 
import sys
import numpy as np

class Graph:
	
	def __init__(self):
		self.nodes = []
		self.edges = []
	def add_node(self,node):
		self.nodes.append(node)
	def add_edge(self,edge):
		self.edges.append(edge)
		edge.vert[0].adjs.append([edge.vert[1],edge.weight])
		edge.vert[1].adjs.append([edge.vert[0],edge.weight])

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
	weight = 0
	def __init__(self,n,v,wt):
		self.no = n
		self.vert = v
		self.weight = wt
	


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
			return True,i.weight
		if i.vert[0].name == node2 and i.vert[1].name == node1:
			return True,i.weight
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
			if j[0] in b:
				cut_size += j[1]
	return cut_size
def klalgo(g):
	g.partition()
	a = g.A
	b = g.B
	# Calculate the cut size
	cut_size = get_cut_size(a,b)
	cut_cost = cut_size
	
	print("#####	Iteration 1")
	print(f'Initial Cut Size: {cut_size}')
	# Calculate the gain of each node
	D = {}
	for i in a:
		D[i.name] = 0
	for i in b:
		D[i.name] = 0
	for i in a:
		for j in i.adjs:
			if j[0] in b:
				D[i.name] += j[1] 
			else:
				D[i.name] -= j[1]
	for i in b:
		for j in i.adjs:
			if j[0] in a:
				D[i.name] += j[1]
			else:
				D[i.name] -= j[1]	
	print('D is \n',D)
	gval = {}
	for i in g.A:
		for j in g.B:
			if i.swapped is False and j.swapped is False:
				cxy = 0
				has_edge,edgewt = isEdgeBetweenNodes(i.name,j.name,g.edges)
				if has_edge:
					cxy = edgewt
				gval[i.name+'-'+j.name] = D[i.name] + D[j.name] - 2*cxy
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

	for i in range(len(a)-1):
		print(f'#####	Iteration {i+2}')
		# Calculate the cut size
		cut_size = get_cut_size(g.A,g.B)


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
					if j[0] in b:
						D_new[i.name] += j[1]
					else:
						D_new[i.name] -= j[1]
		for i in b:
			if i.swapped is False:
				for j in i.adjs:
					if j[0] in a:
						D_new[i.name] += j[1]
					else:
						D_new[i.name] -= j[1]
		gval = {}
		for i in g.A:
			for j in g.B:
				if i.swapped is False and j.swapped is False:
					cxy = 0
					has_edge,edgewt = isEdgeBetweenNodes(i.name,j.name,g.edges)
					if has_edge:
						cxy = edgewt
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
def is_not_integer(s):
	try:
		int(s)
		return False
	except ValueError:
		return True
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
	print(node_list)

	edge_list = []
	with open(edgefile,'r') as ef:
		for i in ef:
			j = i.strip().split('-')	
			edge_list.append(j)
	print(edge_list)
	
	node_obj_dict = {}
	for n in node_list:
		node_obj_dict[n] = Node(n)

	
	for i in node_obj_dict:
		graph.add_node(node_obj_dict[i])
	edge_obj = []
	for i in range(len(edge_list)):	
		e = edge_list[i]
		if is_not_integer(e[2]):
			print(f"ERROR: Weight is not an integer: Edge -> {e[0]}-{e[1]}")
			sys.exit(1)
		
		wt = int(e[2])
		edge_obj.append(Edge(i+1,[node_obj_dict[e[0]],node_obj_dict[e[1]]],wt))

	for i in edge_obj:
		graph.add_edge(i)
	
	return graph


		


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
	print("PARTITION A:")
	for i in graph_new.A:
		print(i.name)
	
	print("PARTITION B:")
	for i in graph_new.B:
		print(i.name)


	nx_graph = graph.to_networkx()

	nx.draw(nx_graph, with_labels=True, node_color="lightgreen", node_size=4000, font_size=12, font_weight="bold", edge_color="blue")
	plt.show()

if __name__ == "__main__":
    __main__()
