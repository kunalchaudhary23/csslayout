from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!2ffaffsfs"


if __name__ == '__main__':
    app.run(debug=True)