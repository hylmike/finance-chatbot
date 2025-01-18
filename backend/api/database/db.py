"""Initial DB config and async instance"""

import os
import contextlib
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    AsyncConnection,
    create_async_engine,
)
from dotenv import load_dotenv
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine

load_dotenv()

DB_ASYNC_URL = os.environ.get("DB_ASYNC_URL")
DB_SYNC_URL = os.environ.get("DB_SYNC_URL")


class Base(DeclarativeBase):
    """Base class for DB models"""

    # setting for execute default func under async scenario
    __mapper_args__ = {"eager_defaults": True}


class DBSessionManager:
    def __init__(self, host: str, engine_kwargs: dict[str, any] = {}):
        self._engin = create_async_engine(host, **engine_kwargs)
        self._sessionmaker = async_sessionmaker(
            autocommit=False, bind=self._engin
        )

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        if self._engin is None:
            raise Exception("Database session manager has not initialized")

        async with self._engin.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("Database session manager has not initialized")

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def close(self):
        if self._engin is None:
            raise Exception("Database session manager has not initialized")

        await self._engin.dispose()
        self._engin = None
        self._sessionmaker = None


db_async_engine = create_async_engine(DB_ASYNC_URL, echo=True)
session_manager = DBSessionManager(DB_ASYNC_URL, {"echo": True})


async def get_db():
    """Get async DB instance with generator"""
    async with session_manager.session() as session:
        yield session


def get_session():
    engin = create_async_engine(DB_ASYNC_URL)
    session_maker = async_sessionmaker(bind=engin)
    return session_maker()


async def create_all_tables():
    """Create all tables based on schema"""
    async with db_async_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


db_sync_engine = create_engine(DB_SYNC_URL)
