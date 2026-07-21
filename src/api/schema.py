from pydantic import BaseModel


class DrugInput(BaseModel):
    cell_type: str
    sm_name: str
    sm_lincs_id: str
    smiles: str
    top_k: int = 20