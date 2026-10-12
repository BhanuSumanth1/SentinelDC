"""Typed, validated physical dependency graph from topology metadata only."""
import json
from pathlib import Path
import networkx as nx

TOPOLOGY=['datacenter_id','rack_id','switch_id','server_id']
TYPES=['DATACENTER','RACK','SWITCH','SERVER']


def node_id(kind,identifier):
    return f'{kind}:{identifier}'


def build_graph(metadata):
    rows=metadata[TOPOLOGY].drop_duplicates()
    if rows.empty or rows.isna().any().any() or rows.eq('').any().any():
        raise ValueError('Topology must contain nonempty infrastructure identifiers.')
    for child,parents in [('server_id',TOPOLOGY[:3]),('switch_id',TOPOLOGY[:2]),('rack_id',['datacenter_id'])]:
        if (rows.groupby(child)[parents].nunique()>1).any().any():
            raise ValueError(f'Conflicting membership or cross-datacenter relationship for {child}.')
    graph=nx.DiGraph()
    for row in rows.sort_values(TOPOLOGY).itertuples(index=False,name=None):
        nodes=[]
        for kind,value in zip(TYPES,row):
            key=node_id(kind,str(value));nodes.append(key)
            graph.add_node(key,node_type=kind,component_id=str(value),datacenter_id=str(row[0]))
        for parent,child in zip(nodes,nodes[1:]):graph.add_edge(parent,child,relationship='CONTAINS')
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Infrastructure relationships must be acyclic.')
    for key,attributes in graph.nodes(data=True):
        expected=0 if attributes['node_type']=='DATACENTER' else 1
        if graph.in_degree(key)!=expected:raise ValueError('Each non-datacenter component must have one parent.')
    return graph


def connected_servers(graph,component):
    if component not in graph:raise ValueError('Unknown infrastructure component.')
    nodes={component}|nx.descendants(graph,component)
    return sorted(graph.nodes[n]['component_id'] for n in nodes if graph.nodes[n]['node_type']=='SERVER')


def blast_radius(graph,component,affected):
    connected=set(connected_servers(graph,component));current=connected&set(affected)
    return {'affected_servers':len(current),'connected_servers':len(connected),
            'potential_additional_exposure':len(connected-current),
            'affected_server_ids':sorted(current),'connected_server_ids':sorted(connected)}


def save_graph(graph,model_dir,report_dir):
    Path(model_dir).mkdir(parents=True,exist_ok=True);Path(report_dir).mkdir(parents=True,exist_ok=True)
    nx.write_graphml(graph,Path(model_dir)/'infrastructure_graph.graphml')
    counts={kind:sum(attrs['node_type']==kind for _,attrs in graph.nodes(data=True)) for kind in TYPES}
    summary={f'number_of_{name}':counts[kind] for name,kind in zip(['datacenters','racks','switches','servers'],TYPES)}
    summary.update(number_of_nodes=graph.number_of_nodes(),number_of_edges=graph.number_of_edges(),
        limitation='Identical rack/switch server membership cannot establish which component caused an incident.')
    (Path(report_dir)/'infrastructure_graph_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
