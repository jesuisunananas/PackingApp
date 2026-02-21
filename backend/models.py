from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import uuid
import json

db = SQLAlchemy()
bcrypt = Bcrypt()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    items = db.relationship('Item', backref='owner', lazy=True)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

class Item(db.Model):
    __tablename__ = 'items'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)

    category = db.Column(db.String(50))
    estimated_value = db.Column(db.Float, default=0.0)

    length = db.Column(db.Float)
    width = db.Column(db.Float)
    height = db.Column(db.Float)

    keypoints_blob = db.Column(db.LargeBinary)

    is_new = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    action_tag = db.Column(db.String(20), default='None')
