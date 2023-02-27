import os, cmd
import openai
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import gtts
import playsound

openai.api_key = 'Your Key'

you_prompt = 'You:'
ai_promt = 'AI:'

start_chat_log = f'''{you_prompt} Hello 
{ai_promt} Hi 
'''

chat_log = None

sampling_rate = 44100 # Hz
threshold = 51 # dB

def get_chat_log(logfile):
    try:
        with open(logfile, 'r') as f:                                                                         
            strt = f.read()    
    except Exception:
        strt = ''
    return strt

def get_model_engine():
    return 'text-davinci-003'

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
    temperature = 0.7
    top = 0.7
    frequency = 1.0
    presence = 1.2
    best = 1
    tokens = 2048
    language = 'German'
    speech = False

    def ask(self, question, chat_log=None):
        if chat_log is None:
            chat_log = start_chat_log
        prompt = f'{chat_log}{you_prompt} {question}\n{ai_promt}'
        try:
            response = openai.Completion.create(
                prompt=prompt, engine=get_model_engine(), stop=[f'\n{you_prompt}'], temperature=self.temperature,
                top_p=self.top, frequency_penalty=self.frequency, presence_penalty=self.presence, best_of=self.best,
                max_tokens=self.tokens )
            answer = response.choices[0].text.strip()
            if self.show_tokens == True:
                print('Tokens: ' + str(response.usage['total_tokens']))
        except Exception:
            answer = 'Oh, I timed out. Try again :-('
        return answer

    def append_interaction_to_chat_log(self, question, answer, chat_log=None):
        if chat_log is None:
            chat_log = start_chat_log
        return f'{chat_log}{you_prompt} {question}\n{ai_promt} {answer}\n'

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
        try:
            args = arg.split()
            if(len(args)==0):
                print(f'tokens: {self.tokens}, temperature: {self.temperature}, top: {self.top}')
                print(f'frequency: {self.frequency}, presence: {self.presence}, best: {self.best}')
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
                else:
                    print('Wrong parameter in command: set.')           
        except Exception:
            print('Syntax error in command: set.')

    def do_cls(self, arg: str):
        print(reset_console(), end='') 

    def do_clear(self, arg: str):
        global chat_log
        chat_log = ''

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
        print(answer)
        if self.speech:
            spgtts = gtts.gTTS(text=answer, lang='de')
            spgtts.save('.\\answer.mp3')
            playsound.playsound(os.getcwd() + '\\answer.mp3')
        chat_log = self.append_interaction_to_chat_log(question, answer, chat_log)
        print(green_text())
        
    def precmd(self, line):
        return line
    
    def voice_rec(self):
        #print('Started recording...')
        audio_chunks = []
        silent = False
        while not silent:
            audio_chunk = sd.rec(int(sampling_rate), dtype='int16', channels=1, blocking=True)
            audio_chunks.append(audio_chunk)
            volume_chunk = volume(audio_chunk)
            #print(f'Volume: {volume_chunk} dB')
            print(make_bar(int(volume_chunk - threshold)), end='')
            if volume_chunk < threshold:
                silent = True
        #print('Stopped recording')
        audio = np.concatenate(audio_chunks)
        sf.write('.\input.wav', audio, sampling_rate)
        model = whisper.load_model('base')
        result = model.transcribe('.\input.wav', fp16=False, language=self.language)
        return result['text']

def volume(audio):
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(np.square(audio)))
    return 20 * np.log10(rms)

if __name__ == '__main__':
    os.system('color')
    print(reset_console()) 
    start_chat_log = get_chat_log('.\log.txt')
    chat_cmd().cmdloop()
