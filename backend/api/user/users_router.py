from fastapi import APIRouter, Depends

from api.user.schemas import User, UserForm
from api.dependencies.db import DBSessionDep
from api.user.services import get_all_users, get_by_id, get_by_name, create_user
from api.dependencies.auth import valid_is_authenticated

users_router = APIRouter()


@users_router.get(
    "/",
    response_model=list[User],
    dependencies=[Depends(valid_is_authenticated)],
)
async def all_users(db: DBSessionDep):
    """Get all users"""
    users = await get_all_users(db)
    return users


@users_router.get(
    "/{user_id}",
    response_model=User,
    dependencies=[Depends(valid_is_authenticated)],
)
async def user_by_id(user_id: int, db: DBSessionDep):
    """Get user by id"""
    user = await get_by_id(db, user_id)
    return user


@users_router.get(
    "/name/{username}",
    response_model=User,
    dependencies=[Depends(valid_is_authenticated)],
)
async def user_by_name(username: str, db: DBSessionDep):
    """Get user by name"""
    user = await get_by_name(db, username)
    return user


@users_router.post(
    "/create",
    response_model=User,
    dependencies=[Depends(valid_is_authenticated)],
)
async def add_user(user_input: UserForm, db: DBSessionDep):
    """Create a new user"""
    user = await create_user(db, user_input)
    return user
