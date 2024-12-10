from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.db.base import Base


class User(Base):
    username: Mapped[str] = mapped_column(unique=True, nullable=False)
    email: Mapped[str] = mapped_column(unique=True, nullable=False)
    password: Mapped[str] = mapped_column(nullable=False)

    datasets: Mapped[list["Dataset"]] = relationship("Dataset", back_populates="owner")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="user")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, username={self.username})"


class Dataset(Base):
    title: Mapped[str] = mapped_column(unique=True, nullable=False)
    file_path: Mapped[str] = mapped_column(unique=True, nullable=False)

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    owner: Mapped["User"] = relationship("User", back_populates="dataset")

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, title={self.title})"


class Prediction(Base):
    result: Mapped[float]

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="prediction")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="prediction")
