from pydantic import BaseModel 

class PersonFacts(BaseModel):
    fact1: str
    fact2: str
    fact3: str 
