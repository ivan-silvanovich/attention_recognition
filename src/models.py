import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, VARCHAR, String, DateTime, LargeBinary, ForeignKey, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.session import sessionmaker

from settings import USERS_PATH

engine = create_engine(f'sqlite:///{os.path.join(USERS_PATH, "users.db")}')
base = declarative_base()
db_session = sessionmaker(bind=engine)


class User(base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(VARCHAR(255), nullable=False, index=True)
    surname = Column(VARCHAR(255), nullable=False, index=True)
    picture = Column(String, nullable=False, unique=True)
    descriptor = Column(LargeBinary, nullable=False)
    created_on = Column(DateTime(), default=datetime.now, nullable=False)
    updated_on = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f'<User(name={self.name}, surname={self.surname})>'
    
    def __str__(self):
        return f'{self.name} {self.surname}'


class Event(base):
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user = Column(Integer, ForeignKey(User.id), nullable=False)
    duration = Column(Float, nullable=False)
    created_on = Column(DateTime(), default=datetime.now, nullable=False)
    updated_on = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f'{self.user} {self.duration}'

    def __str__(self):
        return f'{self.user.name} {self.duration}'


base.metadata.create_all(engine)
