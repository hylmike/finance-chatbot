"""All DB models related to chatbot"""

from enum import StrEnum

from sqlalchemy import Column, Integer, String, DateTime, select, ForeignKey
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from api.database.db import Base
from api.utils.logger import logger


class RoleType(StrEnum):
    SYSTEM = "system"
    AI = "ai"
    HUMAN = "human"
    TOOL = "tool"


class Chat(Base):
    """Chats table to save all chats with AI"""

    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role_type = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created = Column(DateTime, server_default=func.now(), index=True)

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

    @classmethod
    async def find_recent_chat_history(
        cls, db: AsyncSession, user_id: int, limit: int
    ):
        query = await db.scalars(
            select(cls)
            .where(cls.user_id == user_id)
            .order_by(cls.created.desc())
            .limit(limit)
        )
        chat_records = query.all()
        chat_history = {}
        if not chat_records:
            return chat_history
        # If loaded last record is from ai (means answer only), discard it as we need pair of chat messages
        while chat_records[-1].role_type == "ai":
            chat_records.pop()
        chat_records.sort(key=lambda x: x.created)

        index = 0
        while index < len(chat_records):
            human_message = chat_records[index].content
            ai_message = ""
            index += 1
            if (
                index < len(chat_records)
                and chat_records[index].role_type == "ai"
            ):
                ai_message = chat_records[index].content
                index += 1
            chat_history[human_message] = ai_message
            # discard answer without question, normally this should not happen
            while (
                index < len(chat_records)
                and chat_records[index].role_type == "ai"
            ):
                index += 1

        return chat_history


class IngestedFile(Base):
    """ingested_files table to save all ingested files for vector database"""

    __tablename__ = "ingested_files"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_hash = Column(String, nullable=False)
    created = Column(DateTime, server_default=func.now())

    @classmethod
    async def create(cls, db: AsyncSession, **kwargs):
        new_record = cls(**kwargs)
        try:
            db.add(new_record)
            await db.commit()
            await db.refresh(new_record)
        except Exception as e:
            logger.exception(
                f"Failed to insert new record in ingested_files table: {e}"
            )

        return new_record

    @classmethod
    async def find_by_file_hash(cls, db: AsyncSession, file_hash: int):
        query = await db.execute(select(cls).where(cls.file_hash == file_hash))
        return query.scalars().all()
