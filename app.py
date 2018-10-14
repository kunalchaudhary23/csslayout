from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def hello():
	incoming_data = request.args.get()
    return incoming_data


if __name__ == '__main__':
    app.run(debug=True)