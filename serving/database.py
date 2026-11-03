"""Optional PostgreSQL persistence for sentiment prediction history."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from math import ceil
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

LOGGER = logging.getLogger("sentiment_api.database")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DATABASE_CONNECT_TIMEOUT = int(os.getenv("DATABASE_CONNECT_TIMEOUT", "5"))


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_text: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    positive_probability: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True
    )


class DatabaseUnavailable(RuntimeError):
    """Raised when PostgreSQL is not configured or cannot be reached."""


_engine = None
_session_factory: sessionmaker[Session] | None = None


def _get_session_factory() -> sessionmaker[Session]:
    global _engine, _session_factory
    if not DATABASE_URL:
        raise DatabaseUnavailable("DATABASE_URL is not configured")
    if _session_factory is not None:
        return _session_factory
    try:
        connect_args: dict[str, Any] = {}
        if DATABASE_URL.startswith("postgresql"):
            connect_args["connect_timeout"] = DATABASE_CONNECT_TIMEOUT
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
        return _session_factory
    except Exception as exc:
        LOGGER.exception("Could not configure the database engine")
        raise DatabaseUnavailable("Database configuration is invalid") from exc


def ensure_schema() -> None:
    factory = _get_session_factory()
    try:
        if _engine is None:
            raise DatabaseUnavailable("Database engine is unavailable")
        Base.metadata.create_all(_engine)
    except SQLAlchemyError as exc:
        LOGGER.warning("Could not initialize prediction history schema: %s", exc)
        raise DatabaseUnavailable("Database schema is unavailable") from exc


def database_status() -> dict[str, Any]:
    if not DATABASE_URL:
        return {"configured": False, "available": False, "message": "DATABASE_URL is not configured"}
    try:
        factory = _get_session_factory()
        with factory() as session:
            session.execute(select(1))
        return {"configured": True, "available": True}
    except DatabaseUnavailable as exc:
        return {"configured": True, "available": False, "message": str(exc)}
    except SQLAlchemyError:
        return {"configured": True, "available": False, "message": "Database is unavailable"}


def record_prediction(review_text: str, sentiment: str, positive_probability: float) -> int:
    ensure_schema()
    factory = _get_session_factory()
    try:
        with factory() as session:
            record = PredictionRecord(
                review_text=review_text,
                sentiment=sentiment,
                positive_probability=positive_probability,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id
    except SQLAlchemyError as exc:
        LOGGER.warning("Could not save prediction history: %s", exc)
        raise DatabaseUnavailable("Prediction history could not be saved") from exc


def list_predictions(page: int, page_size: int, sentiment: str | None = None) -> dict[str, Any]:
    ensure_schema()
    factory = _get_session_factory()
    try:
        with factory() as session:
            query = select(PredictionRecord)
            count_query = select(func.count()).select_from(PredictionRecord)
            if sentiment:
                query = query.where(PredictionRecord.sentiment == sentiment)
                count_query = count_query.where(PredictionRecord.sentiment == sentiment)
            total = int(session.scalar(count_query) or 0)
            records = session.scalars(
                query.order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            items = [
                {
                    "id": record.id,
                    "text": record.review_text,
                    "sentiment": record.sentiment,
                    "positive_probability": record.positive_probability,
                    "created_at": record.created_at,
                }
                for record in records
            ]
            return {
                "items": items,
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": ceil(total / page_size) if total else 0,
                "sentiment": sentiment,
            }
    except SQLAlchemyError as exc:
        LOGGER.warning("Could not read prediction history: %s", exc)
        raise DatabaseUnavailable("Prediction history is unavailable") from exc
