from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def hello_world():
    return {'message': 'Hello, World!'}

@app.get('/test')
def test():
    return {'message': 'API is running'}
