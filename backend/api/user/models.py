"""All DB models related to user management"""

import logging

from sqlalchemy import Column, Integer, String, DateTime, select
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession

from api.database.db import Base


class User(Base):
    """Users table"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False, index=True)
    password = Column(String, nullable=False)
    created = Column(DateTime, server_default=func.now())
    updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    @classmethod
    async def create(cls, db: AsyncSession, **kwargs):
        try:
            user = cls(**kwargs)
            db.add(user)
            await db.commit()
            await db.refresh(user)
        except Exception as e:
            logging.exception(f"Failed to insert user: {e}")
        return user

    @classmethod
    async def find_all(cls, db: AsyncSession):
        query = await db.execute(select(cls))
        return query.scalars().all()

    @classmethod
    async def find_by_id(cls, db: AsyncSession, id: int):
        user = await db.get(cls, id)
        return user

    @classmethod
    async def find_by_name(cls, db: AsyncSession, name: str):
        user = (
            await db.scalars(select(cls).where(cls.username == name))
        ).first()
        return user
