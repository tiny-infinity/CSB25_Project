
"""
Since the aMAM solver isn't working, I should make good use of the time I have until it
(hopefully) does.

This is code that will take in an adjacency matrix and churn out a .topo file.
"""

def network_to_topo(adjacency_matrix,node_list):
    """
    Format of Adjacency Matrix:
    Rows -> Being Regulated (TO)
    Columns -> Regulating (FROM)
    
    NODE LIST HAS TO BE IN SAME ORDER AS ADJACENCY MATRIX
    """
    topo_list=[]
    if len(node_list)!=len(adjacency_matrix):
        print("ERROR: Adjacency Matrix and Node List are of different length") 
    else:
        for i in range(len(adjacency_matrix)): #regulated
            for j in range(len(adjacency_matrix)): #regulator
                if adjacency_matrix[i][j]==1:
                    interaction=({"Source":node_list[j],
                                      "Target":node_list[i],
                                      "Regulation Type":1})
                elif adjacency_matrix[i][j]==-1:
                    interaction=({"Source":node_list[j],
                                      "Target":node_list[i],
                                      "Regulation Type":2})
                elif adjacency_matrix[i][j]==0:
                    continue
                else:
                    print('ERROR: Adjacency Matrix has invalid values')
                    return None
                topo_list.append(interaction)
    return topo_list

def generate_topo_file(network_name,topo_list):
    """
    Generates .topo file using the created list
    """
    with open(f"{network_name}.topo",'w') as f:
        f.write("Source    Target    Regulation Type")
        f.write("\n")
        for interaction in topo_list:
            f.write(f"{interaction["Source"]}    {interaction["Target"]}    {interaction["Regulation Type"]}")
            f.write("\n")

def save_topo(adj_matrix,node_list,network_name):
    topo_list=network_to_topo(adj_matrix,node_list)
    generate_topo_file(network_name,topo_list)
    print(f"Succesfully generated .topo file for {network_name}")
                    
def topo_to_adj(topo_filename):
    """
    Work the other way 'round too, cause why not?
    Returns adjacency matrix and node list 
    """
    node_list=[]
    topo_list=[]
    with open(topo_filename,'r') as f:
        next(f)
        for line in f:
            src,tgt,reg=line.split()
            if src not in node_list:
                node_list.append(src)
            if tgt not in node_list:
                node_list.append(tgt)
            interaction={"Source":src,
                         "Target":tgt,
                         "Regulation Type":reg}
            topo_list.append(interaction)
    adj_matrix=[[0 for _ in range(len(node_list))] for _ in range(len(node_list))]
    for interaction in topo_list:
        regulated_idx=node_list.index(interaction["Source"])
        regulator_idx=node_list.index(interaction["Target"])
        if interaction["Regulation Type"]=='1':
            adj_matrix[regulated_idx][regulator_idx]=1
        elif interaction["Regulation Type"]=='2':
            adj_matrix[regulated_idx][regulator_idx]=-1
        else:
            print("ERROR: .topo file has invalid values")
            return None
    print("Order of Nodes : ", node_list)
    print("Adjacency Matrix : ")
    for i in range(len(adj_matrix)):
        print(adj_matrix[i])       
    return adj_matrix,node_list
        
four_node_networks=[]

four_node_full = [[0,1,-1,-1],
                  [1,0,-1,-1],
                  [-1,-1,0,1],
                  [-1,-1,1,0]]

four_node_strip1 = [[0,1,-1,-1],
                  [1,0,0,-1],
                  [-1,0,0,1],
                  [-1,-1,1,0]]

four_node_strip2 = [[0,1,-1,0],
                  [1,0,0,-1],
                  [-1,0,0,1],
                  [0,-1,1,0]]

four_node_strip3 = [[0,0,-1,-1],
                  [1,0,-1,-1],
                  [-1,-1,0,1],
                  [-1,-1,0,0]]

four_node_strip4 = [[0,1,-1,0],
                  [1,0,0,-1],
                  [-1,0,0,1],
                  [0,-1,1,0]]

four_node_networks.append(four_node_full)
four_node_networks.append(four_node_strip1)
four_node_networks.append(four_node_strip2)
four_node_networks.append(four_node_strip3)
four_node_networks.append(four_node_strip4)

if __name__ == '__main__':
    for i in range(len(four_node_networks)):
        save_topo(four_node_networks[i],['A1','A2','B1','B2'],f"{i}_four_node")





        
   


            
        

    


