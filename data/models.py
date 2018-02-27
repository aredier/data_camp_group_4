from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, TEXT, ForeignKey, String, FLOAT, BOOLEAN, DATETIME


Base = declarative_base()


class Sentences(Base):

    __tablename__ = 'sentences'

    # review content
    id = Column(Integer, primary_key=True)
    sentence = Column(TEXT, nullable=False)
    review_id = Column(Integer, ForeignKey("reviews.id"),nullable=False)
    review_pos = Column(Integer, nullable=False)



class Reviews(Base):

    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True)
    date_time = Column(DATETIME, nullable=False)
    rating = Column(FLOAT, nullable=True)
    helped = Column(Integer, nullable=True)
    # meta data
    source_review_id = Column(Integer, nullable=True)  # original id of the review on the source it comes from
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)


class Sources(Base):
    """table for the sources"""

    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=True)


class Models(Base):
    """table with the model"""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    brand = Column(String, nullable=False)
    memory_size = Column(Integer, nullable=True)


class Issues(Base):
    """table with the issues of each review"""

    __tablename__ = "issues"

    id = Column(Integer, primary_key=True)
    sentence_id = Column(Integer, ForeignKey("sentences.id"), nullable=False)
    predicted = Column(BOOLEAN, nullable=False)

