from packing.main import packing_with_priors
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from dotenv import load_dotenv
import os
from models import db, User

app = Flask(__name__)
load_dotenv()
CORS(app)

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
jwt = JWTManager(app)

with app.app_context():
    db.create_all()

@app.post('/register')
def register():
    data = request.get_json()

    if User.query.filter_by(email=data['email']).first():
        return jsonify(message="Email already taken"), 400

    if '@' not in data['email']:
        return jsonify(message="Email not valid"), 400

    new_user = User(username=data['username'], email=data['email'])
    new_user.set_password(data['password'])

    db.session.add(new_user)
    db.session.commit()

    return jsonify(message="User registered successfully"), 201

@app.post('/login')
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()

    if not user or not user.check_password(data['password']):
        return jsonify(message="Invalid credentials"), 401
    access_token = create_access_token(identity=user.id)
    return jsonify(access_token=access_token), 200

@app.get('/protected')
@jwt_required()
def protected():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    return jsonify(logged_in_as=user.username), 200

if __name__ == "__main__":
    app.run(debug=True)