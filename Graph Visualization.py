import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, Menu
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from networkx.utils import UnionFind
from networkx.exception import NetworkXNoPath
from networkx.algorithms import minimum_spanning_edges, shortest_path, dijkstra_path, dijkstra_path_length, topological_sort, simple_cycles

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph-based GUI")
        
        self.root.configure(bg="#FAEBD7")

        self.graph_type = tk.StringVar(value="Undirected")
        self.graph = nx.Graph()

        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack()

        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

        self.levels_listbox = tk.Listbox(self.root, width=40, height=10, bg="#ECDCDC")
        self.levels_listbox.pack()


        self.graph_type_radio = ttk.Radiobutton(self.root, text="Undirected", variable=self.graph_type, value="Undirected", command=self.update_graph_type)
        self.graph_type_radio.pack()
        ttk.Radiobutton(self.root, text="Directed", variable=self.graph_type, value="Directed", command=self.update_graph_type).pack()

        style = ttk.Style()
        style.configure("TButton", padding=1, relief="flat", background="#FAEBD7", foreground="black")

        self.add_node_button = ttk.Button(self.root, text="Add Node", command=self.add_node)
        self.add_node_button.pack()

        self.add_edge_button = ttk.Button(self.root, text="Add Edge", command=self.add_edge)
        self.add_edge_button.pack()

        self.add_weight_button = ttk.Button(self.root, text="Add Weight", command=self.add_weight)
        self.add_weight_button.pack()

        self.delete_node_button = ttk.Button(self.root, text="Delete Node", command=self.delete_node)
        self.delete_node_button.pack()

        self.delete_edge_button = ttk.Button(self.root, text="Delete Edge", command=self.delete_edge)
        self.delete_edge_button.pack()

        self.reset_button = ttk.Button(self.root, text="Reset", command=self.reset_algorithm)
        self.reset_button.pack()

        self.apply_menu = Menu(self.root)
        self.root.config(menu=self.apply_menu)

        self.graph_algorithms_menu = Menu(self.apply_menu, tearoff=0)
        self.apply_menu.add_cascade(label="Apply Graph Algorithms", menu=self.graph_algorithms_menu)
        
        self.graph_algorithms_menu.add_command(label="BFS", command=self.apply_bfs)
        self.graph_algorithms_menu.add_command(label="DFS", command=self.apply_dfs)
        self.graph_algorithms_menu.add_command(label="Prim's Algorithm", command=self.apply_prims_algorithm)
        self.graph_algorithms_menu.add_command(label="Kruskal's Algorithm", command=self.apply_kruskal_algorithm)
        self.graph_algorithms_menu.add_command(label="Dijkstra's Algorithm", command=self.apply_dijkstra_algorithm)
        self.graph_algorithms_menu.add_command(label="Topological Sort", command=self.apply_topological_sort)
        self.graph_algorithms_menu.add_command(label="SCC Algorithm", command=self.apply_scc_algorithm)
        self.graph_algorithms_menu.add_command(label="Cycle Detection", command=self.apply_cycle_detection)

        self.draw_graph()

        
    def update_graph_type(self):
        self.graph = nx.Graph() if self.graph_type.get() == "Undirected" else nx.DiGraph()
        self.draw_graph()
        
    def add_node(self):
        node_name = tk.simpledialog.askstring("Add Node", "Enter node name:")
        if node_name:
            self.graph.add_node(node_name)
            self.draw_graph()
            
    def add_edge(self):
        source = tk.simpledialog.askstring("Add Edge", "Enter source node:")
        target = tk.simpledialog.askstring("Add Edge", "Enter target node:")
        
        if source and target:
            self.graph.add_edge(source, target)
            self.draw_graph()
            
    def add_weight(self):
        source = tk.simpledialog.askstring("Add Weight", "Enter source node:")
        target = tk.simpledialog.askstring("Add Weight", "Enter target node:")
        weight = tk.simpledialog.askinteger("Add Weight", "Enter edge weight:")
        
        if source and target and weight and self.graph.has_edge(source, target):
            self.graph[source][target]['weight'] = weight
            self.draw_graph()
        else:
            messagebox.showerror("Error", "Invalid nodes or edge not found.")
            
    def delete_node(self):
        node_name = tk.simpledialog.askstring("Delete Node", "Enter node name:")
        if node_name:
            if node_name in self.graph.nodes:
                self.graph.remove_node(node_name)
                self.draw_graph()
            else:
                messagebox.showerror("Error", "Node not found.")
                
    def delete_edge(self):
        source = tk.simpledialog.askstring("Delete Edge", "Enter source node:")
        target = tk.simpledialog.askstring("Delete Edge", "Enter target node:")
        if source and target:
            if self.graph.has_edge(source, target):
                self.graph.remove_edge(source, target)
                self.draw_graph()
            else:
                messagebox.showerror("Error", "Edge not found.")
    
    def reset_algorithm(self):
        self.reset_node_colors()
        self.canvas.draw() 

    def draw_graph(self):
        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, ax=self.ax, arrows=self.graph_type.get() == "Directed")
        
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        self.canvas.draw()

    def apply_bfs(self):
        start_node = tk.simpledialog.askstring("Apply BFS", "Enter start node:")
        if start_node and start_node in self.graph.nodes:
            self.visualize_bfs(start_node)
        else:
            messagebox.showerror("Error", "Invalid start node.")

    def visualize_bfs(self, start_node):
        visited = set()
        queue = [(start_node, None)]
        added_to_listbox = set()  # To track nodes added to the Listbox

        while queue:
            node, parent = queue.pop(0)
            if node not in visited:
                self.highlight_bfs_node(node, parent, visited, queue, added_to_listbox)
                visited.add(node)
                self.root.update()
                self.root.after(1000)
                neighbors = self.graph.neighbors(node) if self.graph_type.get() == "Undirected" else self.graph.successors(node)
                # Filter out neighbors that have already been visited and those already in the queue
                unvisited_neighbors = [(neighbor, node) for neighbor in neighbors if neighbor not in visited and neighbor not in [item for item, _ in queue]]
                queue.extend(unvisited_neighbors)

        #self.reset_node_colors()


    def highlight_bfs_node(self, node, parent, visited, queue, added_to_listbox):
        colors = []
        for n in self.graph.nodes:
            if n == node and n not in visited:
                colors.append("red")  # Current node
                visited.add(node)  # Mark the node as visited
            elif n == parent:
                colors.append("grey")  # Parent node
            elif n in (self.graph.neighbors(node) if self.graph_type.get() == "Undirected" else self.graph.successors(node)) and n not in visited:
                colors.append("yellow")  # Adjacent nodes that are not visited
            else:
                colors.append("grey")  # Unvisited nodes

        # Update the queue in the Listbox to show the current visiting node and its adjacent unvisited nodes
        queue_text = f"{node}"
        if queue:
            neighbors_to_add = [neighbor for neighbor, _ in queue if neighbor not in visited and neighbor not in added_to_listbox]
            neighbors_to_add = [n for n in neighbors_to_add if n not in queue]  # Filter out nodes that are already in the queue
            if neighbors_to_add:
                queue_text += ", " + ", ".join(neighbors_to_add)

        # Check if the node is not already in the Listbox, then add it
        if node not in added_to_listbox:
            self.levels_listbox.insert(tk.END, queue_text)
            self.levels_listbox.see(tk.END)  # Scroll to the latest entry
            self.levels_listbox.update()
            added_to_listbox.add(node)

        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color=colors, ax=self.ax, arrows=self.graph_type.get() == "Directed")
        self.canvas.draw()



    def apply_dfs(self):
        start_node = tk.simpledialog.askstring("Apply DFS", "Enter start node:")
        if start_node and start_node in self.graph.nodes:
            visited = set()
            self.visualize_dfs(start_node, visited)

            # After visiting all reachable nodes, visit disconnected nodes
            for node in self.graph.nodes:
                if node not in visited:
                    self.visualize_dfs(node, visited)
        else:
            messagebox.showerror("Error", "Invalid start node.")


    def visualize_dfs(self, start_node, visited):
        stack = [(start_node, None)]
        stack_nodes = []  # Maintain a separate stack to keep track of the nodes in the order they were visited
        backtrack_stack = []  # Stack for backtracking

        while stack:
            node, parent = stack.pop()
            if node in visited:
                continue  # Skip nodes that are already visited
            stack_nodes.append(node)  # Push the current node to the stack_nodes
            self.highlight_dfs_node(node, parent, visited, stack_nodes, backtrack_stack)
            visited.add(node)
            self.root.update()
            self.root.after(1000)

            adjacent_unvisited_nodes = []
            if self.graph_type.get() == "Undirected":
                adjacent_unvisited_nodes = [neighbor for neighbor in self.graph.neighbors(node) if neighbor not in visited]
            else:
                adjacent_unvisited_nodes = [neighbor for neighbor in self.graph.successors(node) if neighbor not in visited]

            if not adjacent_unvisited_nodes:
                # No more unvisited adjacent nodes, backtrack
                while stack_nodes:
                    top_node = stack_nodes[-1]
                    if top_node == start_node and not adjacent_unvisited_nodes:
                        break  # Reached the starting node, stop backtracking
                    if (self.graph_type.get() == "Undirected" and all(neighbor in visited for neighbor in self.graph.neighbors(top_node))) or \
                            (self.graph_type.get() == "Directed" and all(neighbor in visited for neighbor in self.graph.successors(top_node))):
                        # Mark backtracked node as pink and pop from the stack_nodes
                        backtrack_stack.append(stack_nodes.pop())
                    else:
                        break

            # Push unvisited neighbors onto the stack
            stack.extend((neighbor, node) for neighbor in adjacent_unvisited_nodes)


    def highlight_dfs_node(self, node, parent, visited, stack_nodes, backtrack_stack):
        colors = []

        for n in self.graph.nodes:
            if n == node and n not in visited:
                colors.append("red")  # Current node
                visited.add(node)  # Mark the node as visited
            elif n == parent:
                colors.append("grey")  # Parent node
            elif n in (self.graph.neighbors(node) if self.graph_type.get() == "Undirected" else self.graph.successors(node)) and n not in visited:
                colors.append("yellow")  # Unvisited adjacent nodes
            elif n in backtrack_stack:
                colors.append("pink")  # Backtracked node
            else:
                colors.append("grey")  # Unvisited nodes

        # Update the Listbox one line at a time
        stack_text = ", ".join(stack_nodes)
        self.levels_listbox.insert(tk.END, stack_text)
        self.levels_listbox.see(tk.END)  # Scroll to the latest entry
        self.levels_listbox.update()

        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color=colors, ax=self.ax, arrows=self.graph_type.get() == "Directed")
        self.canvas.draw()

    def apply_prims_algorithm(self):
        if len(self.graph.nodes) == 0:
            messagebox.showerror("Error", "No nodes in the graph.")
            return
            
        if self.graph_type.get() == "Directed":
            messagebox.showerror("Error", "Prim's algorithm does not work for directed graphs.")
            return

        # Check if all edges have weights
        for u, v in self.graph.edges:
            if 'weight' not in self.graph[u][v]:
                messagebox.showerror("Error", "Please add weights to all edges before applying Prim's algorithm.")
                return
        
        # Proceed with the algorithm
        self._apply_prims_algorithm()

    def _apply_prims_algorithm(self):
        # Get the minimum spanning tree using Prim's Algorithm
        min_spanning_tree = nx.minimum_spanning_tree(self.graph)
            
        # Get the edges in the order they appear in the minimum spanning tree
        edges_to_highlight = list(min_spanning_tree.edges())
        
        for i, edge in enumerate(edges_to_highlight):
            self.highlight_prims_edges(edges_to_highlight[:i+1])
            self.root.update() 
            self.root.after(1000)  

    def highlight_prims_edges(self, edges):
        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, ax=self.ax)

        # Draw all the edges that have been highlighted so far with a different color
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color='red', width=2)

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        edge_labels = {edge: edge_labels[edge] for edge in edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        self.canvas.draw()


    def apply_kruskal_algorithm(self):
        if len(self.graph.nodes) == 0:
            messagebox.showerror("Error", "No nodes in the graph.")
            return
        
        if self.graph_type.get() == "Directed":
            messagebox.showerror("Error", "Kruskal's algorithm does not work for directed graphs.")
            return
         # Check if all edges have weights
        for u, v in self.graph.edges:
            if 'weight' not in self.graph[u][v]:
                messagebox.showerror("Error", "Please add weights to all edges before applying Kruskal algorithm.")
                return
        
        # Proceed with the algorithm
        self._apply_kruskal_algorithm()

    def _apply_kruskal_algorithm(self):
        
        # Get the minimum spanning tree using Kruskal's Algorithm
        min_spanning_edges_gen = minimum_spanning_edges(self.graph, data=False)
        edges_to_highlight = list(min_spanning_edges_gen)
        
        for i, edge in enumerate(edges_to_highlight):
            self.highlight_kruskal_edges(edges_to_highlight[:i+1])
            self.root.update() 
            self.root.after(1000)
    
    def highlight_kruskal_edges(self, edges):
        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, ax=self.ax, arrows=self.graph_type.get() == "Directed")

        # Draw all the edges that have been highlighted so far with a different color
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color='red', width=2)

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        edge_labels = {edge: edge_labels[edge] for edge in edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        self.canvas.draw()

    def apply_dijkstra_algorithm(self):
        if len(self.graph.nodes) == 0:
            messagebox.showerror("Error", "No nodes in the graph.")
            return
        
        if any(self.graph[u][v].get('weight', 0) < 0 for u, v in self.graph.edges):
            messagebox.showerror("Error", "Dijkstra's algorithm doesn't work for graphs with negative edge weights.")
            return
        
        source = tk.simpledialog.askstring("Apply Dijkstra's Algorithm", "Enter source node:")
        target = tk.simpledialog.askstring("Apply Dijkstra's Algorithm", "Enter target node:")
        
        if source and target and source in self.graph.nodes and target in self.graph.nodes:
            self.highlight_dijkstra_path(source, target)
        else:
            messagebox.showerror("Error", "Invalid source or target node.")
            
    def highlight_dijkstra_path(self, source, target):
        try:
            shortest_path_nodes = nx.dijkstra_path(self.graph, source, target)
        except NetworkXNoPath:
            messagebox.showinfo("Info", f"There is no path from {source} to {target}.")
            return
        
        shortest_path_edges = [(shortest_path_nodes[i], shortest_path_nodes[i + 1]) for i in range(len(shortest_path_nodes) - 1)]
        
        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, ax=self.ax, arrows=self.graph_type.get() == "Directed")

        nx.draw_networkx_edges(self.graph, pos, edgelist=shortest_path_edges, edge_color='red', width=2)

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        path_edge_labels = {edge: edge_labels.get(edge, edge_labels.get(edge[::-1])) for edge in shortest_path_edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=path_edge_labels)

        self.canvas.draw()

    def apply_topological_sort(self):
        if self.graph_type.get() == "Undirected":
            messagebox.showinfo("Info", "Topological sort is only applicable to directed graphs.")
            return
        
        if not nx.is_directed_acyclic_graph(self.graph):
            messagebox.showinfo("Info", "Graph contains cycles. Topological sort is not possible.")
            return
        
        # Get a copy of the graph for step-by-step sorting
        step_graph = self.graph.copy()
        while step_graph.nodes():
            next_node = self.get_next_topological_node(step_graph)
            if next_node is None:
                break
            
            self.highlight_topological_sort(next_node)
            self.root.update()
            self.root.after(1000)
            
            step_graph.remove_node(next_node)
        
        self.reset_algorithm()

    def get_next_topological_node(self, graph):
        indegree = dict(graph.in_degree())
        for node, degree in indegree.items():
            if degree == 0:
                return node
        return None

    def highlight_topological_sort(self, node):
        self.reset_node_colors()
        colors = ["lightblue" for _ in self.graph.nodes]
        node_index = list(self.graph.nodes).index(node)
        colors[node_index] = "red"
        
        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color=colors, ax=self.ax, arrows=self.graph_type.get() == "Directed")
        
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        self.canvas.draw()

    def apply_scc_algorithm(self):
        if self.graph_type.get() == "Undirected":
            messagebox.showinfo("Info", "Strongly Connected Components is applicable only to directed graphs.")
            return

        # Find the strongly connected components using Kosaraju's Algorithm
        sccs = list(nx.strongly_connected_components(self.graph))

        # Highlight each SCC one by one
        for scc in sccs:
            self.highlight_scc(scc)
            self.root.update()
            self.root.after(1000)

        self.reset_algorithm()

    def highlight_scc(self, scc):
        self.reset_node_colors()
        colors = ["lightblue" for _ in self.graph.nodes]
        
        for node in scc:
            node_index = list(self.graph.nodes).index(node)
            colors[node_index] = "red"

        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color=colors, ax=self.ax, arrows=self.graph_type.get() == "Directed")

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        self.canvas.draw()

    def apply_cycle_detection(self):
        if self.graph_type.get() == "Directed":
            cycles = list(nx.simple_cycles(self.graph))
        else:
            cycles = list(nx.cycle_basis(self.graph))

        for cycle in cycles:
            self.highlight_cycle(cycle)
            self.root.update()
            self.root.after(1000)

        self.reset_algorithm()

    def highlight_cycle(self, cycle):
        self.reset_node_colors()
        colors = ["lightblue" for _ in self.graph.nodes]
        
        for node in cycle:
            node_index = list(self.graph.nodes).index(node)
            colors[node_index] = "red"

        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color=colors, ax=self.ax, arrows=self.graph_type.get() == "Directed")

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        self.canvas.draw()


    def reset_algorithm(self):
        self.reset_node_colors()
        self.canvas.draw()

    def reset_node_colors(self):
        colors = ["lightblue" for _ in self.graph.nodes]
        self.ax.clear()
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color=colors, ax=self.ax, arrows=self.graph_type.get() == "Directed")
        
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        self.canvas.draw()
        
        self.levels_listbox.delete(0, tk.END)  # Clear all entries in the Listbox


if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()
