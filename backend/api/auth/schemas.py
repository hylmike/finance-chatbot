from pydantic import BaseModel, ConfigDict


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str


class LoginForm(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    username: str
    password: str
