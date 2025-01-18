from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import NoResultFound
from fastapi import HTTPException
from passlib.context import CryptContext

from .schemas import User, UserForm
from .models import User as UserModel


pw_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pw_context.hash(password)


async def get_all_users(db: AsyncSession) -> list[User]:
    users = await UserModel.find_all(db)
    return users


async def get_by_name(db: AsyncSession, name: str) -> User:
    try:
        user = await UserModel.find_by_name(db=db, name=name)
    except NoResultFound:
        raise HTTPException(status_code=404, detail=f"user {name} not found")
    return user


async def get_by_id(db: AsyncSession, id: int) -> User:
    try:
        user = await UserModel.find_by_id(db=db, id=id)
    except NoResultFound:
        raise HTTPException(
            status_code=404, detail=f"user with id {id} not found"
        )
    return user


async def create_user(db: AsyncSession, user_input: UserForm) -> User:
    hashed_password = hash_password(user_input.password)
    user_input.__dict__.update({"password": hashed_password})
    try:
        user = await UserModel.create(db, **user_input.model_dump())
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Faied to create user {user_input.username}",
        )
    return user
