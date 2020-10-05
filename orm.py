import os
from dataclasses import dataclass

from contextlib import closing

import numpy as np
from datetime import datetime

from sqlalchemy import MetaData,Table,Column,Integer,String,Date,create_engine,ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import create_session,relationship,sessionmaker

from hashlib import md5

engine = create_engine("sqlite:///:memory:")
metadata = MetaData(bind=engine)
Base = declarative_base(bind=engine, metadata = metadata)
Session  = sessionmaker(bind=engine)
metadata.create_all()

"""
Types
"""

@dataclass
class Daterange:
    start: datetime.date
    end: datetime.date

@dataclass
class Coordinate:
    x: float 
    y: float 

@dataclass
class Rect:
    tl: Coordinate
    tr: Coordinate
    br: Coordinate
    bl: Coordinate

"""
Continuous dimensions
"""
# X
# Y
# T

"""
Nominal dimensions
"""
class Feature(Base):
    __tablename__ = "feature"
    id = Column("id",Integer,primary_key=True)
    name = Column("name",String(60),nullable=False)
    tensors = relationship("Tensor")

    def __repr__(self):
        return f"{self.name}"
