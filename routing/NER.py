from gliner import GLiNER
from typing import List, Callable

from routing.router import Router
from routing.synset import OSMSemanticBridge


class GlinerAStarRouter(Router):
    """
    NER-based router that uses GLiNER to extract semantic entities from prompts
    and maps them to OSM tags via the OSMSemanticBridge.

    Args:
        graph:      OSMnx/NetworkX MultiDiGraph of the road network.
        weights:    Two-element list [avoid_penalty, prefer_reward].
                    e.g. [25.0, 0.01].
        bridge:     An OSMSemanticBridge instance for semantic-to-OSM mapping.
        model_name: HuggingFace model ID for GLiNER entity extraction.
    """

    def __init__(self, graph, weights: List[float], bridge: OSMSemanticBridge,
                 model_name="urchade/gliner_medium-v2.1"):
        super().__init__(graph)
        self.bridge = bridge
        self.gliner_model = GLiNER.from_pretrained(model_name)
        self.labels = ["road_type_to_avoid", "amenity_required"]
        self.weights = weights

    def _build_weight_func(self, prompt: str) -> Callable[[int, int, dict], float]:
        """
        Extract entities from the prompt with GLiNER, map them to OSM tags
        via the semantic bridge, and return a weight function that penalizes
        or rewards edges matching those tags.
        """
        entities = self.gliner_model.predict_entities(prompt, self.labels, threshold=0.55)

        trigger_map = {}
        for ent in entities:
            ent_text = ent['text'].lower()
            category_mapped_matches = self.bridge.get_osm_synsets(ent_text)
            if category_mapped_matches:
                trigger_map[ent_text] = {
                    "mapped_categories": category_mapped_matches,
                    "label": ent['label']
                }

        def weight_func(u, v, edge_dict):
            data = list(edge_dict.values())[0] if isinstance(edge_dict, dict) else edge_dict[0]
            base_cost = data.get('length', 1.0)

            if not trigger_map:
                return base_cost

            mult = 1.0
            for ent_text, info in trigger_map.items():
                is_match = False
                for category, target_values in info["mapped_categories"].items():
                    actual_val = data.get(category)
                    if actual_val:
                        current_vals = actual_val if isinstance(actual_val, list) else [str(actual_val)]
                        if any(val in target_values for val in current_vals):
                            is_match = True
                            break

                if is_match:
                    if info['label'] == "road_type_to_avoid":
                        mult *= self.weights[0]
                    elif info['label'] == "amenity_required":
                        mult *= self.weights[1]

            return base_cost * mult

        return weight_func