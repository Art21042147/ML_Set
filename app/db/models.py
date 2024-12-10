from datetime import datetime
from sqlalchemy import ForeignKey, DateTime, func
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.db.base import Base


class User(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True, nullable=False)
    email: Mapped[str] = mapped_column(unique=True, nullable=False)
    password: Mapped[str] = mapped_column(nullable=False)

    datasets: Mapped[list["Dataset"]] = relationship("Dataset", back_populates="owner")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="user")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, username={self.username})"


class Dataset(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(unique=True, nullable=False)
    file_path: Mapped[str] = mapped_column(unique=True, nullable=False)

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    owner: Mapped["User"] = relationship("User", back_populates="dataset")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, title={self.title})"


class Prediction(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    result: Mapped[float]
    created_on: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="prediction")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="prediction")
