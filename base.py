from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_wtf import FlaskForm
from wtforms import (StringField, BooleanField, DateTimeField, 
                        RadioField, SelectField, TextAreaField, 
                        SubmitField)
from wtforms.validators import DataRequired
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import subprocess
import os
import secrets
import magenta

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mykey'
app.secret_key = "My_Key"

# ----- 작사 관련 함수 -----
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(sequence, max_length, top_k, top_p, repetition_penalty, num_return_sequences):
    model_path = "./ch1models"
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}.', return_tensors='pt')
    print(ids, model)
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=num_return_sequences,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in final_outputs]


# ----- 필요한 페이지 -----
@app.route('/')
def routePage():
    return render_template("home.html")

@app.route('/compose')
def composePage():
    return render_template('compose.html')

@app.route('/write')
def writePage():
    return render_template('write.html')

@app.route('/help')
def helpPage():
    return render_template('help.html')


# ----- 작사 기능 구현 -----
@app.route('/writing', methods=['POST'])
def write():
    # 입력값을 설정해둔 id로 받아오기
    keyword = str(request.form['keyword'])
    max_Length = int(request.form['maxLength'])
    top_k = int(request.form['top_k'])
    top_p = float(request.form['top_p'])
    repetitionpenalty = float(request.form['repetitionpenalty'])
    num_return_sequences = 1
    print(keyword, max_Length, top_k, top_p, repetitionpenalty)
    
    # 결과물 만들어내기
    generated_lyrics = generate_text(keyword,
                                    max_Length,
                                    top_k,
                                    top_p,
                                    repetitionpenalty,
                                    num_return_sequences,
                                    )
    writeList = generated_lyrics[0].split('\n')
    
    # HTML 가져오기
    return render_template("writeResult.html", result = writeList)

# ----- 작곡 기능 구현 -----
@app.route('/composing', methods=['POST'])
def compose():
    # 고유 session 생성
    user_identifier = session.get('user_id', secrets.token_urlsafe(16))
    session['user_id'] = user_identifier
    output_dir = "static/song/" + session['user_id']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    melody_Length = request.form['melodyLength']
    first_Melody = request.form['firstMelody']
    
    # 명령어 CODE 생성
    first_sentence = "python magenta/magenta/models/melody_rnn/melody_rnn_generate.py --config=attention_rnn --run_dir=DL_in_Music/FF_rnn --output_dir="
    dir_path = output_dir + " --num_outputs=1 --num_steps="
    second_sentence =  " --hparams=\"batch_size=64,rnn_layer_sizes=[64,64]\" --primer_melody=\"["
    third_sentence = "]\""
    
    total_sentence = first_sentence + dir_path + melody_Length + second_sentence + first_Melody + third_sentence
    
    os.system(total_sentence)
    
    # SESSIOM 값과 같은 폴더에서 가장 최근 생성된 MIDI의 경로 생성
    path = "C:/Users/user_name/OneDrive/바탕 화면/Contents_2/" + output_dir
    file_list = os.listdir(path)
    file_name = file_list[len(file_list) - 1]
    song_path = "static/song/" + session['user_id'] + "/" + file_name
    
    # HTML 가져오기
    return render_template('songResult.html', file_name = song_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)