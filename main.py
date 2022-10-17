from flask import Flask, abort, request, send_file
from tempfile import NamedTemporaryFile
from processor import Processor
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
proc = Processor()

@app.route('/')
@cross_origin()
def hello_world(): 
    return 'ditmehuytrau!'

@app.route('/answer', methods=['POST'])
@cross_origin()
def handler():
    if not request.files:
        abort(400)
    _, handle = request.files.items()[0]
    temp = NamedTemporaryFile()
    handle.save(temp)
    audio_path = proc.run(temp.name)
    return send_file(audio_path)

@app.route('/getlog', methods=['GET'])
@cross_origin()
def get_log():
    return proc.get_conversation_log()

@app.route('/textans', methods=['POST'])
@cross_origin()
def textans():
    if not request.files:
        abort(400)
    _, handle = list(request.files.items())[0]
    temp = NamedTemporaryFile()
    handle.save(temp)
    resp = proc.run(temp.name, aud=False)
    return {'error' : resp[0], 'correct' : resp[1], "ans" : resp[2]}

@app.route('/geterrorlog', methods=['GET'])
@cross_origin()
def get_error_log():
    return proc.get_conversation_log(error=True)