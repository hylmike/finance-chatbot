from typing import Annotated
from datetime import timedelta
import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from dotenv import load_dotenv

from .schemas import Token
from .services import authenticate_user, create_access_token
from api.dependencies.db import DBSessionDep

load_dotenv()

auth_router = APIRouter()


@auth_router.post("/token", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: DBSessionDep
):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expires_in = os.environ.get("ACCESS_TOKEN_EXPIRES_IN", "15")
    token_expires_in = timedelta(minutes=int(expires_in))
    data = {"sub": user.username}
    access_token = create_access_token(data, token_expires_in)

    return Token(access_token=access_token, token_type="bearer")
