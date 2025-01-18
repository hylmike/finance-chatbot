"""All DB models related to chatbot"""

from enum import StrEnum

from sqlalchemy import Column, Integer, String, DateTime, select, ForeignKey
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from api.database.db import Base
from api.utils.logger import logger


class RoleType(StrEnum):
    SYSTEM = "SYSTEM"
    AI = "AI"
    HUMAN = "HUMAN"
    TOOL = "TOOL"


class Chat(Base):
    """Chats table to save all chats with AI"""

    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role_type = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created = Column(DateTime, server_default=func.now())

    @classmethod
    async def create(cls, db: AsyncSession, **kwargs):
        new_chat = cls(**kwargs)
        try:
            db.add(new_chat)
            await db.commit()
            await db.refresh(new_chat)
        except Exception as e:
            logger.exception(f"Failed to insert chat: {e}")

        return new_chat

    @classmethod
    async def find_by_userid(cls, db: AsyncSession, user_id: int):
        query = await db.execute(select(cls).where(cls.user_id == user_id))
        return query.scalars().all()

    @classmethod
    async def find_recent_human_records(
        cls, db: AsyncSession, user_id: int, limit: int
    ):
        query = await db.scalars(
            select(cls)
            .where(cls.user_id == user_id, cls.role_type == RoleType.HUMAN)
            .order_by(cls.created.desc())
            .limit(limit)
        )
        return query.all()
