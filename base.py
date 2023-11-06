from flask import Flask, render_template, request, jsonify, session, redirect, url_for
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
import magenta

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mykey'

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


# ----- 내부 링크 연결 -----
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


# ----- 작곡 기능 구현 -----
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
    
    # 결과값 반환
    return render_template("writeResult.html", result = writeList)

@app.route('/composing', methods=['POST'])
def compose():
    import magenta
    melody_Length = int(request.form['melodyLength'])
    first_Melody = int(request.form['firstMelody'])
    num_return_mid = int(request.form['songResult'])

    os.system("python magenta/magenta/models/melody_rnn/melody_rnn_generate.py --config=attention_rnn --run_dir=DL_in_Music/FF_rnn --output_dir=DL_in_Music/FF_attention_6464 --num_outputs=1 --num_steps=640 --hparams=\"batch_size=64,rnn_layer_sizes=[64,64]\" --primer_melody=\"[57]\"")
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)