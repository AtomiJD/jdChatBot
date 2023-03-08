import os, cmd
from datetime import datetime
import openai
import sounddevice as sd
import soundfile as sf
import numpy as np
import gtts
import playsound
import json
from pygments import highlight
#from pygments.style import Style
#from pygments.token import Token
from pygments.lexers import Python3Lexer
from pygments.formatters import Terminal256Formatter

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_model_engine(engine='gpt-3.5-turbo'):
    if engine == 'gpt-3.5-turbo':
        return 'gpt-3.5-turbo'
    else:
        return 'text-davinci-003'

def reset_engine():
    global system_role
    global assistant_role
    global user_role
    global start_chat_log
    global chat_log
    if get_model_engine() == 'gpt-3.5-turbo':
        system_role = 'system'
        assistant_role = 'assistant'
        user_role = 'user'
        start_chat_log = [{"role": system_role, "content": "Hello"}, {"role": assistant_role, "content": "Hi"}]
    else:
        you_prompt = 'You:'
        ai_prompt = 'AI:'
        start_chat_log = f'''{you_prompt} Hello 
        {ai_prompt} Hi 
        '''
    chat_log = None

sampling_rate = 44100 
threshold = 51 

def get_chat_log(logfile):
    try:
        if get_model_engine() == 'gpt-3.5-turbo':
            with open(logfile, 'r', encoding='utf-8') as f:
                strf = f.read()
                strf = strf.replace('{current_date}', str(datetime.now().strftime('%d.%m.%Y')))
                strt = json.loads(strf)
        else:
            with open(logfile, 'r', encoding='utf-8') as f:
                strt = f.read().format(datetime.now().strftime('%d.%m.%Y'))
    except Exception:
        print('Cannot load chat log.')
        strt = ''
    return strt

def reset_console():
    return '\033c\x1b[2J'

def green_text():
    return '\x1b[0;32;40m'

def blue_text():
    return '\x1b[0;34;40m'

def make_bar(size):
    return '\x1b[s\x1b[1;1H' + ' ' * 80 + '\x1b[1;1H' + '|' * size + '\x1b[u'

class chat_cmd(cmd.Cmd):
    intro = blue_text() + 'jdChat with engine: ' + get_model_engine() + ' ready!\n'
    intro += 'Enter your commands (help, quit, cls) in lower case please.\n'
    intro += 'Enter your message and press return:'
    prompt = green_text() + '> '
    show_tokens = False
    temperature = 0.9
    top = 0.7
    frequency = 1.0
    presence = 1.2
    best = 1
    tokens = 2048
    language = 'German'
    speech = False
    highlightcode = False

    def ask(self, question, chat_log=None):
        if get_model_engine() == 'gpt-3.5-turbo':
            prompt = question
            global start_chat_log
            chat_dict = {}
            chat_dict['role'] = user_role
            chat_dict['content'] = question
            start_chat_log.append(chat_dict)          
        else:
            if chat_log is None:
                chat_log = start_chat_log
            prompt = f'{chat_log}{you_prompt} {question}\n{ai_prompt}'
        try:
            response = openai.Moderation.create(
                input=prompt  )
            output = response['results'][0]
            if output['flagged'] == True:
                answer = 'Your request is violating OpenAI\'s content policy.'
            else:
                if get_model_engine() == 'gpt-3.5-turbo':
                    response = openai.ChatCompletion.create(
                    model=get_model_engine(), 
                    messages=start_chat_log
                    )
                    answer = response['choices'][0]['message']['content']
                else:
                    response = openai.Completion.create(
                        prompt=prompt, engine=get_model_engine(), stop=[f'\n{you_prompt}',f'\n{ai_prompt}'], temperature=self.temperature,
                        top_p=self.top, frequency_penalty=self.frequency, presence_penalty=self.presence, best_of=self.best,
                        max_tokens=self.tokens )
                    answer = response.choices[0].text.strip()
            if self.show_tokens == True:
                print('Tokens: ' + str(response.usage['total_tokens']))
        except Exception:
            answer = 'Oh, I timed out. Try again :-('
        return answer

    def concat_chat_log(self, question, answer, chat_log=None):
        if chat_log is None:
            chat_log = start_chat_log
        return f'{chat_log}{you_prompt} {question}\n{ai_prompt} {answer}\n'

    def do_help(self, arg: str):
        try:
            with open('./help.txt', 'r') as f:                                                                         
                strt = f.read()    
            print(strt)
        except Exception:
            print('Please enter a question and press return.')
            print('To exit type quit')
            print('Type cls to clear chat window')
    
    def do_quit(self, arg: str):
        print('Thanks for playing jdChat :-)')
        return True

    def do_set(self, arg: str):
        global you_prompt
        global ai_prompt
        try:
            args = arg.split()
            if(len(args)==0):
                print(f'tokens: {self.tokens}, temperature: {self.temperature}, top: {self.top}')
                print(f'frequency: {self.frequency}, presence: {self.presence}, best: {self.best}')
                print(f'language: {self.language}, highlightcode: {self.highlightcode}, prompt: {you_prompt} {ai_prompt}')
                print(f'speech: {self.speech}')
            else:
                if(args[0]=='tokens'):
                    self.show_tokens = True if args[1]=='on' else False
                elif(args[0]=='speech'):
                    self.speech = True if args[1]=='on' else False
                elif(args[0]=='temperature'):
                    self.temperature = float(args[1])
                elif(args[0]=='top'):
                    self.top = float(args[1])
                elif(args[0]=='frequency'):
                    self.frequency = float(args[1])
                elif(args[0]=='presence'):
                    self.presence = float(args[1])
                elif(args[0]=='best'):
                    self.best = float(args[1])
                elif(args[0]=='tokens'):
                    self.tokens = int(args[1])
                elif(args[0]=='language'):
                    self.language = args[1]
                if(args[0]=='syntax'):
                    self.highlightcode = True if args[1]=='on' else False
                elif(args[0]=='prompt'):
                    you_prompt = args[1]
                    if (len(args)==2):
                        ai_prompt = args[2]
                else:
                    print('Wrong parameter in command: set.')           
        except Exception:
            print('Syntax error in command: set.')

    def do_cls(self, arg: str):
        print(reset_console(), end='') 

    def do_list(self, arg: str):
        if get_model_engine() == 'gpt-3.5-turbo':
            global start_chat_log
            print(start_chat_log )
        else:
            global chat_log
            print(chat_log) 

    def do_clear(self, arg: str):
        if get_model_engine() == 'gpt-3.5-turbo':
            global start_chat_log
            start_chat_log = []
        else:
            global chat_log
            chat_log = ''

    def do_save(self, arg: str):
        global chat_log
        args = arg.split()
        if(len(args)==0):
            strn = 'log_{0}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        else:
            strn = args[0]
        strex = '.json',  6 if get_model_engine() == 'gpt-3.5-turbo' else '.txt', 5
        if len(strn) < strex[1] or not strn[-(strex[1]-1):].lower() == strex[0]:
            strn += strex[0]
        try:
            with open(strn, 'wb') as f:
                if get_model_engine() == 'gpt-3.5-turbo':
                    f.write(json.dumps(start_chat_log).encode('utf-8'))
                else:                                            
                    f.write(chat_log.encode('utf-8'))
        except Exception:
            print('Cannot write log: ' + strn)

    def emptyline(self):
        print('Please enter something.')

    def do_record(self, arg: str):
        strt = self.voice_rec()
        print(self.prompt + strt)
        self.default(strt)

    def default(self, line: str):
        global chat_log
        question = line
        print(blue_text(), end='')
        answer = self.ask(question, chat_log)
        if self.highlightcode:
                print(blue_text()+highlight(answer, Python3Lexer(), Terminal256Formatter(style='github-dark')))
        else:
            print(answer)
        if self.speech:
            spgtts = gtts.gTTS(text=answer, lang='de')
            spgtts.save('.\\answer.mp3')
            playsound.playsound(os.getcwd() + '\\answer.mp3')
        if get_model_engine() == 'gpt-3.5-turbo':
            chat_dict = {}
            chat_dict['role'] = assistant_role
            chat_dict['content'] = answer
            start_chat_log.append(chat_dict)
        else:            
            chat_log = self.concat_chat_log(question, answer, chat_log)
        print(green_text())
        
    def precmd(self, line):
        return line
    
    def voice_rec(self):
        audio_chunks = []
        silent = False
        while not silent:
            audio_chunk = sd.rec(int(sampling_rate), dtype='int16', channels=1, blocking=True)
            audio_chunks.append(audio_chunk)
            volume_chunk = volume(audio_chunk)
            print(make_bar(int(volume_chunk - threshold)), end='')
            if volume_chunk < threshold:
                silent = True
        audio = np.concatenate(audio_chunks)
        sf.write('.\\input.wav', audio, sampling_rate)
        file = open('.\\input.wav', "rb")
        transcription = openai.Audio.transcribe("whisper-1", file)
        return transcription['text']

def volume(audio):
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(np.square(audio)))
    return 20 * np.log10(rms)

if __name__ == '__main__':
    reset_engine()
    os.system('color')
    print(reset_console()) 
    if get_model_engine() == 'gpt-3.5-turbo':
        start_chat_log = get_chat_log('.\log.json')
    else:
        start_chat_log = get_chat_log('.\log.txt')
    chat_cmd().cmdloop()
