from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import declared_attr

from config import config


class Base(DeclarativeBase):
    __abstract__ = True

    metadata = MetaData(
        naming_convention=config.db.naming_convention,
    )

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
