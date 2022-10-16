from flask import Flask, abort, request, send_file
from tempfile import NamedTemporaryFile
from processor import Processor

app = Flask(__name__)
proc = Processor()

@app.route('/')
def hello_world(): 
    return 'ditmehuytrau!'

@app.route('/answer', methods=['POST'])
def handler():
    if not request.files:
        abort(400)
    _, handle = request.files.items()[0]
    temp = NamedTemporaryFile()
    handle.save(temp)
    audio_path = proc.run(temp.name)
    return send_file(audio_path)

@app.route('/getlog', methods=['GET'])
def get_log():
    return proc.get_conversation_log()

@app.route('/geterrorlog', methods=['GET'])
def get_error_log():
    return proc.get_conversation_log(error=True)
