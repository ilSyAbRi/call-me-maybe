from pydantic import BaseModel

class Prompt(BaseModel):
    prompt: str

class Parameter(BaseModel):
    type: str

class ReturnType(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType