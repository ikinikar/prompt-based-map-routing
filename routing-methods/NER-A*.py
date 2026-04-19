import networkx as nx
import osmnx as ox
from gliner import GLiNER

class GlinerAStarRouter:
    def __init__(self, graph, model_name="urchade/gliner_medium-v2.1"):
        self.graph = graph
        self.gliner_model = GLiNER.from_pretrained(model_name)
        # Actor Model: all-MiniLM-L6-v2
        self.st_router = SentenceTransformer('all-MiniLM-L6-v2')
        self.labels = ["road_type_to_avoid", "amenity_required"]
        
        # Pre-cache unique tags to prevent model-looping during Dijkstra
        print("Enriching graph semantic index...")
        unique_tags = set()
        for _, _, data in graph.edges(data=True):
            for k in ['name', 'highway', 'amenity', 'shop', 'leisure', 'landuse']:
                v = data.get(k)
                if v and str(v) != 'nan':
                    unique_tags.add(str(v).lower())
        
        self.tag_list = list(unique_tags)
        self.tag_embeddings = self.st_router.encode(self.tag_list, convert_to_tensor=True)

    def find_route(self, start_node, end_node, prompt: str):
        entities = self.gliner_model.predict_entities(prompt, self.labels, threshold=0.55)
        if not entities:
            return nx.dijkstra_path(self.graph, start_node, end_node, weight='length')

        # Map GLiNER entities to OSM tags via 0.85 similarity threshold
        entity_texts = [ent['text'].lower() for ent in entities]
        entity_embs = self.st_router.encode(entity_texts, convert_to_tensor=True)
        cos_sim = util.cos_sim(entity_embs, self.tag_embeddings)
        
        trigger_map = {}
        for i, ent in enumerate(entities):
            # 0.85 Threshold prevents 'semantic noise' (verified by your 0.9 trial)
            match_indices = torch.where(cos_sim[i] > 0.85)[0]
            trigger_map[ent['text'].lower()] = {
                "tags": [self.tag_list[idx] for idx in match_indices],
                "label": ent['label']
            }

        def weight_func(u, v, edge_dict):
            data = list(edge_dict.values())[0] if isinstance(edge_dict, dict) else edge_dict[0]
            base_cost = data.get('length', 1.0)
            
            # Aggregate all metadata for the road segment
            edge_meta = " ".join([str(data.get(k, '')).lower() for k in ['name', 'highway', 'amenity', 'shop', 'leisure']])
            
            mult = 1.0
            for ent_text, info in trigger_map.items():
                # Fuzzy Check: Does the edge have a tag semantically linked to the entity?
                if any(t in edge_meta for t in info['tags']) or ent_text in edge_meta:
                    if info['label'] == "road_type_to_avoid":
                        mult *= 10000.0 # Aggressive penalty
                    elif info['label'] == "amenity_required":
                        mult *= 0.000001 # Aggressive reward
            return base_cost * mult

        try:
            return nx.dijkstra_path(self.graph, start_node, end_node, weight=weight_func)
        except:
            return None