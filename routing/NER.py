import networkx as nx
import osmnx as ox
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from typing import Literal, List, Dict
from routing.synset import OSMSemanticBridge

class GlinerAStarRouter:
    def __init__(self, graph, weights: List[float], bridge: OSMSemanticBridge, model_name="urchade/gliner_medium-v2.1"):
        self.graph = graph
        self.bridge = bridge  # Dependency Injection of the Schema-Grounded Bridge
        self.gliner_model = GLiNER.from_pretrained(model_name)
        self.labels = ["road_type_to_avoid", "amenity_required"]
        self.weights = weights # e.g., [25.0, 0.01]

    def find_route(self, start_node, end_node, prompt: str, algorithm: Literal["dijkstra", "astar"] = "dijkstra"):
        # 1. NLP Extraction: Identify semantic constraints in the prompt
        entities = self.gliner_model.predict_entities(prompt, self.labels, threshold=0.55)
        
        # 2. Semantic Translation: Map intent to technical OSM categories/values via Schema
        trigger_map = {}
        for ent in entities:
            ent_text = ent['text'].lower()
            # Bridge returns category-mapped dict: {'highway': ['motorway_link'], ...}
            category_mapped_matches = self.bridge.get_osm_synsets(ent_text)
            
            if category_mapped_matches:
                trigger_map[ent_text] = {
                    "mapped_categories": category_mapped_matches,
                    "label": ent['label']
                }

        # 3. The Attribute-Grounded Weight Function
        def weight_func(u, v, edge_dict):
            # Standardize edge data access for MultiGraphs
            data = list(edge_dict.values())[0] if isinstance(edge_dict, dict) else edge_dict[0]
            base_cost = data.get('length', 1.0)
            
            if not trigger_map:
                return base_cost
            
            mult = 1.0
            for ent_text, info in trigger_map.items():
                is_match = False
                
                # Check actual edge attributes against mapped schema categories
                for category, target_values in info["mapped_categories"].items():
                    actual_val = data.get(category)
                    
                    if actual_val:
                        # Handle OSM data (strings or lists of values)
                        current_vals = actual_val if isinstance(actual_val, list) else [str(actual_val)]
                        
                        # Set inclusion check for precision
                        if any(val in target_values for val in current_vals):
                            is_match = True
                            break
                
                # Apply penalty/reward multipliers
                if is_match:
                    if info['label'] == "road_type_to_avoid":
                        mult *= self.weights[0]
                    elif info['label'] == "amenity_required":
                        mult *= self.weights[1]
            
            return base_cost * mult

        # 4. Restored Working Heuristic
        def heuristic(n, target):
            n_data, t_data = self.graph.nodes[n], self.graph.nodes[target]
            return ox.distance.great_circle(n_data['y'], n_data['x'], t_data['y'], t_data['x'])

        # 5. Pathfinding Execution
        try:
            if algorithm == "astar":
                return nx.astar_path(self.graph, start_node, end_node, 
                                   heuristic=heuristic, weight=weight_func)
            else:
                return nx.dijkstra_path(self.graph, start_node, end_node, weight=weight_func)
        except Exception as e:
            print(f"Routing Error ({algorithm}): {e}")
            return None