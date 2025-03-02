import os
import jwt
from datetime import datetime, timedelta, UTC

from dotenv import load_dotenv
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from api.user.schemas import User
from api.user import services as user_services

load_dotenv()

ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET", "")
ACCESS_TOKEN_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

pw_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pw_context.verify(plain_password, hashed_password)


async def authenticate_user(
    db: AsyncSession, username: str, password: str
) -> User | None:
    user = await user_services.get_by_name(db, username)
    if not user:
        return None
    if not verify_password(password, user.password):
        return None

    return user


def decode_jwt(token: str) -> dict:
    return jwt.decode(
        token, ACCESS_TOKEN_SECRET, algorithms=[ACCESS_TOKEN_ALGORITHM]
    )


def create_access_token(data: dict, expires_in: timedelta | None) -> str:
    to_encode = data.copy()
    if expires_in:
        expire = datetime.now(UTC) + expires_in
    else:
        expire = datetime.now(UTC) + timedelta(minutes=60)
    to_encode.update({"exp": expire})

    access_token = jwt.encode(
        to_encode, ACCESS_TOKEN_SECRET, algorithm=ACCESS_TOKEN_ALGORITHM
    )

    return access_token
