from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, Text
from database.database import Base

class Receipt(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    prediction = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
