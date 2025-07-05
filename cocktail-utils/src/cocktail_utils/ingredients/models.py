import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class IngredientMatch:
    brand: Optional[str]
    specific_type: Optional[str]
    category: str
    confidence: float
    source: str  # 'dictionary', 'llm', 'manual'


@dataclasses.dataclass
class IngredientUsage:
    ingredient_name: str
    recipe_count: int
    sample_recipes: List[str]
