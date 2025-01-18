from typing import Annotated
from jwt.exceptions import InvalidTokenError

from fastapi import Depends, HTTPException, status

from api.auth.services import oauth2_scheme, decode_jwt
from .db import DBSessionDep
from api.user.schemas import User
from api.user.services import get_by_name


async def get_current_user_from_token(
    token: Annotated[str, Depends(oauth2_scheme)], db: DBSessionDep
) -> User:
    auth_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="You are not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_jwt(token)
        username = payload.get("sub")
        if not username:
            raise auth_exception
    except InvalidTokenError:
        raise auth_exception
    user = await get_by_name(db, username)
    if not user:
        raise auth_exception

    return user


CurrentUserDep = Annotated[User, Depends(get_current_user_from_token)]


async def valid_is_authenticated(current_user: CurrentUserDep) -> User:
    """Auth dependency with access token validation"""
    return current_user
