from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.database.db import get_db

DBSessionDep = Annotated[AsyncSession, Depends(get_db)]
