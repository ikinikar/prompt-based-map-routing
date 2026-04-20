import torch
from sentence_transformers import util

class OSMSemanticBridge:
    def __init__(self, tag_schema, st_model, threshold=0.80):
        """
        Initializes the bridge using a structured schema rather than raw graph crawling.
        
        Args:
            tag_schema (dict): The dictionary containing 'continuous' and 'discrete' OSM keys.
            st_model: The SentenceTransformer instance.
            threshold (float): Similarity floor for a 'synset' match.
        """
        self.st_model = st_model
        self.threshold = threshold
        self.schema = tag_schema
        
        # We focus on discrete categories as they represent the 'Synsets' for NLP mapping
        self.categories = list(tag_schema["discrete"].keys())
        
        # 1. Targeted Indexing: Build a semantic map for each category in the schema
        print(f"Bridge: Building Schema-Grounded Index for categories: {self.categories}")
        self.category_indices = {}
        
        for cat in self.categories:
            raw_values = tag_schema["discrete"][cat]
            # Clean values for better embedding (e.g., 'motorway_link' -> 'motorway link')
            clean_values = [str(v).lower().replace('_', ' ') for v in raw_values]
            
            # Pre-cache vectors for this specific category
            embeddings = self.st_model.encode(clean_values, convert_to_tensor=True)
            
            self.category_indices[cat] = {
                "original": raw_values,
                "clean": clean_values,
                "embeddings": embeddings
            }

    def get_osm_synsets(self, entity_text: str):
        """
        Maps a natural language entity to specific OSM keys and values.
        
        Returns:
            dict: { 'highway': ['motorway_link', 'trunk_link'], 'junction': ['roundabout'] }
        """
        entity_vec = self.st_model.encode(entity_text.lower(), convert_to_tensor=True)
        mapped_synsets = {}
        
        for cat, index in self.category_indices.items():
            # Calculate similarity against this category's possible values
            scores = util.cos_sim(entity_vec, index["embeddings"])[0]
            
            # Find all matches above the threshold
            match_indices = torch.where(scores > self.threshold)[0]
            
            if len(match_indices) > 0:
                mapped_synsets[cat] = [index["original"][i] for i in match_indices]
        
        return mapped_synsets