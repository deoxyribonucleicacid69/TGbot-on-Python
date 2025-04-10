import telebot
import sqlite3
from telebot import types
import speech_recognition as sr
import datetime
import soundfile as sf
import numpy as np
from PROVERKI import proverka_str
from PROVERKI import proverka_chisla
import re

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


import pymorphy2

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('punkt')
df1 = pd.read_csv('diagnozes.csv', skipinitialspace=True, sep=';')
logfile = str(datetime.date.today()) + '.log' # формируем имя лог-файла
bot = telebot.TeleBot("6934422448:AAEicqhp-c-7Hr4GXVOmZG2fTR_ecBL1Irs")
token = "6934422448:AAEicqhp-c-7Hr4GXVOmZG2fTR_ecBL1Irs"
nltk.download('punkt')  # Загрузка токенизатора для NLTK

morph = pymorphy2.MorphAnalyzer()

age = None
height = None
ves = None
poison = None
sex = None
oper = None
chron = None
fff=False

df = pd.read_csv("final.csv")

stop_words = stopwords.words("russian")



def extract_main_meaning(sentence):
    # Удаление не русских символов из предложения
    sentence = re.sub(r'[^А-Яа-яЁё\s]', '', sentence)

    # Токенизируем предложение
    tokens = word_tokenize(sentence, language='russian')

    # Лемматизируем слова и удаляем "стоп-слова" (например, местоимения, предлоги и союзы)
    lemmatized_words = [morph.parse(token)[0].normal_form for token in tokens if
                        morph.parse(token)[0].tag.POS not in {'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}]

    # Выводим основной смысл текста
    a = ' '.join(lemmatized_words)
    return a
def data_preprocessing(review):

    # tokenization
    tokens = word_tokenize(review)

    # stop words removal
    review = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]

    review = ' '.join(review)

    return review


df['обработанные_жалобы'] = df['жалобы'].apply(lambda review: data_preprocessing(review))

df = df.drop(["жалобы"], axis=1)

data = df.copy()
y = data['специальность_врача'].values
data.drop(['специальность_врача'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

print(f"Train data: {X_train.shape, y_train.shape}")
print(f"Test data: {X_test.shape, y_test.shape}")

vectorizer = TfidfVectorizer(min_df=10)

X_train_review_tfidf = vectorizer.fit_transform(X_train['обработанные_жалобы'])
X_test_review_tfidf = vectorizer.transform(X_test['обработанные_жалобы'])

print('X_train_review_tfidf shape: ', X_train_review_tfidf.shape)
print('X_test_review_tfidf shape: ', X_test_review_tfidf.shape)

clf = LogisticRegression(penalty='l2')
clf.fit(X_train_review_tfidf, y_train)

y_pred = clf.predict(X_test_review_tfidf)
print('Test Accuracy: ', accuracy_score(y_test, y_pred))



@bot.message_handler(commands=["start"])
def start(message):
    try:
        conn = sqlite3.connect("bd1.sql")
        cur = conn.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS users (id int auto_increment primary key,sex text,AGE int,HEIGHT int'
            ',VES int ,POISON text,OPER text,CHRON text)')
        conn.commit()
        cur.close()
        conn.close()
        markup = types.ReplyKeyboardMarkup()
        btn1 = types.KeyboardButton("Заполнить профиль")
        markup.row(btn1)
        bot.send_message(message.chat.id, f"Здравствуйте {message.from_user.first_name} {message.from_user.last_name}",
                         reply_markup=markup)
        bot.send_message(message.chat.id,
                         "Мы создали нашего бота ,чтобы люди могли узнать по своей истории болезни и жалобам"
                         " что с ними не так ")
        bot.send_message(message.chat.id, "Поэтому вам придеться рассказть небольшие секретики о вас", )
        bot.send_message(message.chat.id,
                         "вам необходимо заполнить профиль, для этого нажмите на кнопку'Заполнить профиль' ", )
        bot.register_next_step_handler(message, prof)
    except Exception:
        bot.send_message(message.chat.id,
                         "Произошла какая-то ошибка, попробуйте перезапутсить бота командной '/start', или о"
                         "братитесь в "
                         "техподдержку:@fucking_kurwa_man")


@bot.message_handler(commands=["info"])
def main(message):
    bot.send_message(message.chat.id, message)


def prof(message):
    bot.send_message(message.chat.id, "Введите свой пол:", )
    bot.register_next_step_handler(message, pol)


def pol(message):
    global sex
    sex = message.text
    if proverka_str(sex):
        bot.send_message(message.chat.id, "Хорошо,Введите свой возраст")
        bot.register_next_step_handler(message, Age)
    else:
        bot.send_message(message.chat.id, "Введите корректные данные (букавки)")
        bot.register_next_step_handler(message, pol)


def Age(message):
    global age
    age = message.text
    if proverka_chisla(age):
        bot.send_message(message.chat.id, "Отлично,теперь введите свой рост: ")
        bot.register_next_step_handler(message, Height)
    else:
        bot.send_message(message.chat.id, "Введите корректные данные (циферки)")
        bot.register_next_step_handler(message, Age)


def Height(message):
    global height
    height=message.text
    if proverka_chisla(height):
        bot.send_message(message.chat.id, "Замечательно, теперь введите ваш вес", )
        bot.register_next_step_handler(message, Ves)
    else:
        bot.send_message(message.chat.id, "Введите корректные данные (циферки)")
        bot.register_next_step_handler(message, Height)


def Ves(message):
    global ves
    ves = message.text
    if proverka_chisla(ves):
        bot.send_message(message.chat.id,
                         "Отлично теперь введите болезни которыми вы переболели!")
        bot.register_next_step_handler(message, OPER)
    else:
        bot.send_message(message.chat.id, "Введите корректные данные (циферки)")
        bot.register_next_step_handler(message,Ves )

def OPER(message):
    global poison
    poison = message.text
    if proverka_str(poison):
        poison=message.text
        a = poison
        a = a.lower()
        a = a.replace(',', '')
        a = a.replace('-', '')
        a = a.replace('.', '')
        preprocessed_input = data_preprocessing(a)
        test = np.array([preprocessed_input])
        test_series = pd.Series(test)
        test_tfidf = vectorizer.transform(test_series)
        poison = test_tfidf
        bot.send_message(message.chat.id,"Теперь вам необходимо ввести ваши жалобы")
        bot.register_next_step_handler(message, CHRON)
    else:
        bot.send_message(message.chat.id, "Введите корректные данные (циферки)")
        bot.register_next_step_handler(message, OPER)



@bot.message_handler(content_types=["text",'voice'])
def CHRON(message):
    global fff
    global oper
    if message.content_type == 'voice':
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open('new_file.ogg', 'wb') as new_file:
            new_file.write(downloaded_file)
            new_file.close()
        data, samplerate = sf.read('new_file.ogg')
        sf.write('new_file.wav', data, samplerate)
        oper = speech_to_text('new_file.wav')
        if proverka_str(oper):
            s1=oper
            for i in df1['симптомы']:
                s2 = i
                tfidf_vectorizer = TfidfVectorizer()

                # Преобразование предложений в TF-IDF векторы
                tfidf_matrix = tfidf_vectorizer.fit_transform([oper, s2])

                # Вычисление косинусного сходства между векторами
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
                if cosine_sim[0][1] > 0.075:

                    fff=True
                    break
                l = extract_main_meaning(s1)
                k = extract_main_meaning(s2)

            bot.send_message(message.chat.id,
                             "Если у вас имеются хронические заболевания то введите их.\n (через запятую)\nЕсли же у вас их нету то введите 'Нет':")
            bot.register_next_step_handler(message, lie)
        else:
            bot.send_message(message.chat.id, "Введите корректные данные (букавки)")
            bot.register_next_step_handler(message, CHRON)
    else:
        oper=message.text
        s1 = oper
        if proverka_str(oper):
            s1 = oper
            for i in df1['симптомы']:
                s2 = i
                tfidf_vectorizer = TfidfVectorizer()

                # Преобразование предложений в TF-IDF векторы
                tfidf_matrix = tfidf_vectorizer.fit_transform([oper, s2])

                # Вычисление косинусного сходства между векторами
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
                if cosine_sim[0][1] > 0.075:
                    fff = True
                    break
                l = extract_main_meaning(s1)
                k = extract_main_meaning(s2)

            bot.send_message(message.chat.id,
                             "Если у вас имеются хронические заболевания то введите их.\n (через запятую)\nЕсли же у вас их нету то введите 'Нет':")
            bot.register_next_step_handler(message, lie)
        else:
            bot.send_message(message.chat.id, "Введите корректные данные (букавки)")
            bot.register_next_step_handler(message, CHRON)


def lie(message):
    global poison
    global age
    global height
    global ves
    global sex
    global chron
    global oper
    global fff
    chron = message.text
    if int(age)>=18:
        if proverka_str(chron):

            conn = sqlite3.connect("bd1.sql")
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO users (SEX,AGE,HEIGHT,VES,POISON,OPER,CHRON) VALUES("%s","%s","%s","%s","%s","%s","%s")' % (
                sex, age, height, ves, poison, oper, chron))
            conn.commit()
            cur.close()
            conn.close()
            markup = types.ReplyKeyboardMarkup()
            btn2 = types.KeyboardButton("Перейти в профиль")
            btn69 = types.KeyboardButton("Техподдрежка")
            btn6 = types.KeyboardButton("Больница рядом")
            btn8 = types.KeyboardButton("Редактировать профиль")
            markup.row(btn2,btn8)
            markup.row(btn6, btn69)


            if fff:
                bot.send_message(message.chat.id,
                                 f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves}\nВаш специалист: {clf.predict(poison)[0].upper()} \nВЫ НЕМЕДЛЕННО ДОЛЖНЫ ОБРАТИТЬСЯ К ВРАЧУ В ВАШЕМ ГОРОДЕ\nЖалбоы: {oper} \nХронические заболевания: {chron}\n"
                                 f"Позвоние в 103 или 112",
                                 reply_markup=markup)
                bot.send_message(message.chat.id,
                                 "Москва и Московская:+7 (495) 431-70-04  \nСанкт-Петербург: (812) 336 33 33 \nБарнаул (3852) 63 68 38 \nБрянск: +7 (4832) 32-34-26\nВладикавказ: +7 (8672) 77-40-10 \nВолгоград: +7 (8442) 59-10-12 \nВыксаг: +7 (831) 773-93-03 \nИжевск: (3412) 91-20-03 \nНижневартовск:  (3466) 29-88-00 \nНягань: (34672) 5-93-40 \nПермь: (342) 2 150 630 \nРостов-на-Дону: +7 (863) 280-00-33 \nТуапсе: +7 (3412) 91-20-03 \nСаранск: +7 (8342) 37-00-70 \nУфа: +7 (347) 225-21-61")
                bot.register_next_step_handler(message, profile)
            else:
                bot.send_message(message.chat.id,
                                f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves}\nВаш специалист: {clf.predict(poison)[0].upper()}\nВЫ МОЖЕТЕ ОБРАТИТЬСЯ К ВРАЧУ В ЛЮБОЕ ВРЕМЯ \nЖалобы: {oper} \nХронические заболевания: {chron}",
                                reply_markup=markup)
                bot.register_next_step_handler(message, profile)
        else:
            bot.send_message(message.chat.id, "Введите корректные данные (букавки)")
            bot.register_next_step_handler(message, lie)
    else:
        bot.send_message(message.chat.id, "Ваш возраст меньше 18, поэтому я не могу вам рекомендовать врача.\nСходите к педиатру вашего района с родителями.\n"
                                          "Чтобы изменить возраст, начните работу бота командой '/start'")


def profile(message):
    global age
    global height
    global ves
    global sex
    global chron
    global oper
    try:
        if message.text == "Перейти в профиль":
            bot.send_message(message.chat.id,
                             f"Вот ваши данные {message.from_user.first_name} {message.from_user.last_name}:\nВаш пол:{sex} \nВозраст:{age} лет\nРост: {height} cантиметров \nВес: {ves} кг \nВаш специалист: {clf.predict(poison)[0].upper()}\nПеренесённые операции:{oper}\nХронические заболевания:{chron}")
            bot.register_next_step_handler(message, profile)
        elif message.text == "Техподдрежка":
            bot.send_message(message.chat.id,
                             "Глава проекта:https://t.me/da_robertosik | @da_robertosik \n Разработчик телеграмм-бота: https://t.me/fucking_kurwa_man | @fucking_kurwa_man \n Разработчик нейросети: https://t.me/lootally | @lootally \nОтветственные по АД: https://t.me/da_robertosik | @da_robertosik и https://t.me/dea1ler | @dea1ler \nТехподдрежка может отвечать в течение 6-9 рабочих дней")
            bot.register_next_step_handler(message, profile)

        elif message.text == "Редактировать профиль":
            markup = types.InlineKeyboardMarkup(row_width=4)
            btn1 = types.InlineKeyboardButton("Возраст", callback_data="voz")
            btn2 = types.InlineKeyboardButton("Вес", callback_data="veS")
            btn3 = types.InlineKeyboardButton("Рост", callback_data="rost")
            btn4 = types.InlineKeyboardButton("Пол", callback_data="pl")
            btn5 = types.InlineKeyboardButton("Жалобы", callback_data="pois")
            btn6 = types.InlineKeyboardButton("Операции", callback_data="operation")
            btn7 = types.InlineKeyboardButton("Хронические заболеваний", callback_data="chronik")
            btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
            markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
            bot.send_message(message.chat.id,
                             f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                             reply_markup=markup)
            bot.register_next_step_handler(message, profile)
        elif message.text == "Больница рядом":
            bot.send_message(message.chat.id,
                             "Вам необоходимо поделится своей геолокацией \nВведиет свое место положение(Белгород, Бульвар Юности)")
            bot.register_next_step_handler(message, karta)



    except Exception:
        bot.send_message(message.chat.id,
                         "Произошла какая-то ошибка, попробуйте перезапутсить бота командной '/start', или обратитесь в техподдержку:@fucking_kurwa_man")


@bot.callback_query_handler(func=lambda call: True)
def redakcia(call):
    if call.data == "voz":
        bot.send_message(call.message.chat.id, "Введите новый возраст:")
        bot.register_next_step_handler(call.message, vozz)
    elif call.data == "veS":

        bot.send_message(call.message.chat.id, "Введите новый вес:")
        bot.register_next_step_handler(call.message, vess)
    elif call.data == "rost":
        bot.send_message(call.message.chat.id, "Введите новый рост:")
        bot.register_next_step_handler(call.message, rost)
    elif call.data == "pl":
        bot.send_message(call.message.chat.id, "Введите новый пол(не ламинат или плитка, а мужской или женский):")
        bot.register_next_step_handler(call.message, poll)
    elif call.data == "pois":
        bot.send_message(call.message.chat.id, "Введите жалобы:")
        bot.register_next_step_handler(call.message, poisons)
    elif call.data == "operation":
        bot.send_message(call.message.chat.id, "Введите перенесенные операции:")
        bot.register_next_step_handler(call.message, operacii)
    elif call.data == "chronik":
        bot.send_message(call.message.chat.id, "Введите хронические заболевания:")
        bot.register_next_step_handler(call.message, chroniki)
    elif call.data == "nazad":
        bot.send_message(call.message.chat.id, "Редактирование профиля закончилось")
        bot.register_next_step_handler(call.message, profile)


def vozz(message):
    conn = sqlite3.connect("bd1.sql")
    cur = conn.cursor()
    conn.commit()
    global age
    age = message.text
    if int(age)>=18:
        "UPDATE bd1 SET AGE = age  WHERE id == chat.id"
        markup = types.InlineKeyboardMarkup(row_width=4)
        btn1 = types.InlineKeyboardButton("Возраст", callback_data="voz")
        btn2 = types.InlineKeyboardButton("Вес", callback_data="veS")
        btn3 = types.InlineKeyboardButton("Рост", callback_data="rost")
        btn4 = types.InlineKeyboardButton("Пол", callback_data="pl")
        btn5 = types.InlineKeyboardButton("Жалобы", callback_data="pois")
        btn6 = types.InlineKeyboardButton("Операции", callback_data="operation")
        btn7 = types.InlineKeyboardButton("Хронические заболеваний", callback_data="chronik")
        btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
        markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
        bot.send_message(message.chat.id, "Возраст успешно изменен")
        bot.send_message(message.chat.id,
                         f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                         reply_markup=markup)
        bot.register_next_step_handler(message, redakcia)
        cur.close()
        conn.close()
    else:
        bot.send_message(message.chat.id,"Простите но ваш возраст меньше 18 лет, поэтому мы не можем вам рекомендовать "
                                         "врача.\nСходите с вашими родителями(Законными представителями)в больницу/детскую поликлинику и обратитесь к педиатру")
        bot.register_next_step_handler(message,profile)


def vess(message):
    conn = sqlite3.connect("bd1.sql")
    cur = conn.cursor()
    conn.commit()
    global ves
    ves = message.text
    "UPDATE bd1 SET VES = ves  WHERE id == chat.id"
    markup = types.InlineKeyboardMarkup(row_width=4)
    btn1 = types.InlineKeyboardButton("Возраст", callback_data="voz")
    btn2 = types.InlineKeyboardButton("Вес", callback_data="veS")
    btn3 = types.InlineKeyboardButton("Рост", callback_data="rost")
    btn4 = types.InlineKeyboardButton("Пол", callback_data="pl")
    btn5 = types.InlineKeyboardButton("Жалобы", callback_data="pois")
    btn6 = types.InlineKeyboardButton("Операции", callback_data="operation")
    btn7 = types.InlineKeyboardButton("Хронические заболеваний", callback_data="chronik")
    btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
    bot.send_message(message.chat.id, "Вес успешно изменен")
    bot.send_message(message.chat.id,
                     f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                     reply_markup=markup)
    bot.register_next_step_handler(message, redakcia)
    cur.close()
    conn.close()


def rost(message):
    conn = sqlite3.connect("bd1.sql")
    cur = conn.cursor()
    conn.commit()
    global height
    height = message.text
    "UPDATE bd1 SET HEIGHT = height  WHERE id == chat.id"
    markup = types.InlineKeyboardMarkup(row_width=4)
    btn1 = types.InlineKeyboardButton("Возраст", callback_data="voz")
    btn2 = types.InlineKeyboardButton("Вес", callback_data="veS")
    btn3 = types.InlineKeyboardButton("Рост", callback_data="rost")
    btn4 = types.InlineKeyboardButton("Пол", callback_data="pl")
    btn5 = types.InlineKeyboardButton("Жалобы", callback_data="pois")
    btn6 = types.InlineKeyboardButton("Операции", callback_data="operation")
    btn7 = types.InlineKeyboardButton("Хронические заболеваний", callback_data="chronik")
    btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
    bot.send_message(message.chat.id,
                     f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                     reply_markup=markup)
    bot.send_message(message.chat.id, "Рост успешно изменен")
    bot.register_next_step_handler(message, redakcia)
    cur.close()
    conn.close()


def poll(message):
    conn = sqlite3.connect("bd1.sql")
    cur = conn.cursor()
    conn.commit()
    global sex
    sex = message.text
    "UPDATE bd1 SET SEX = sex  WHERE id == chat.id"
    markup = types.InlineKeyboardMarkup(row_width=4)
    btn1 = types.InlineKeyboardButton("Возраст", callback_data="voz")
    btn2 = types.InlineKeyboardButton("Вес", callback_data="veS")
    btn3 = types.InlineKeyboardButton("Рост", callback_data="rost")
    btn4 = types.InlineKeyboardButton("Пол", callback_data="pl")
    btn5 = types.InlineKeyboardButton("Жалобы", callback_data="pois")
    btn6 = types.InlineKeyboardButton("Операции", callback_data="operation")
    btn7 = types.InlineKeyboardButton("Хронические заболеваний", callback_data="chronik")
    btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
    bot.send_message(message.chat.id,
                     f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                     reply_markup=markup)
    bot.send_message(message.chat.id, "Пол успешно изменен")
    bot.register_next_step_handler(message, redakcia)
    cur.close()
    conn.close()


def poisons(message):
    conn = sqlite3.connect("bd1.sql")
    cur = conn.cursor()
    conn.commit()
    global poison
    poison = message.text
    "UPDATE bd1 SET POISON = poison  WHERE id == chat.id"
    markup = types.InlineKeyboardMarkup(row_width=4)
    btn1 = types.InlineKeyboardButton("Изменить возраст", callback_data="voz")
    btn2 = types.InlineKeyboardButton("Изменить вес", callback_data="veS")
    btn3 = types.InlineKeyboardButton("Изменить рост", callback_data="rost")
    btn4 = types.InlineKeyboardButton("Изменить пол", callback_data="pl")
    btn5 = types.InlineKeyboardButton("Изменить жалобы", callback_data="pois")
    btn6 = types.InlineKeyboardButton("Изменить перенесенные операции", callback_data="operation")
    btn7 = types.InlineKeyboardButton("Изменить списко хронических заболеваний", callback_data="chronik")
    btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
    bot.send_message(message.chat.id,
                     f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                     reply_markup=markup)
    bot.send_message(message.chat.id, ":Жалобы успешно изменены")
    bot.register_next_step_handler(message, redakcia)
    cur.close()
    conn.close()


def chroniki(message):
    conn = sqlite3.connect("bd1.sql")
    cur = conn.cursor()
    conn.commit()
    global chron
    chron = message.text
    "UPDATE bd1 SET CHRON = chron  WHERE id == chat.id"
    markup = types.InlineKeyboardMarkup(row_width=4)
    btn1 = types.InlineKeyboardButton("Возраст", callback_data="voz")
    btn2 = types.InlineKeyboardButton("Вес", callback_data="veS")
    btn3 = types.InlineKeyboardButton("Рост", callback_data="rost")
    btn4 = types.InlineKeyboardButton("Пол", callback_data="pl")
    btn5 = types.InlineKeyboardButton("Жалобы", callback_data="pois")
    btn6 = types.InlineKeyboardButton("Операции", callback_data="operation")
    btn7 = types.InlineKeyboardButton("Хронические заболевания", callback_data="chronik")
    btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
    bot.send_message(message.chat.id,
                     f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                     reply_markup=markup)
    bot.send_message(message.chat.id, "Хронические заболевания успешно изменены")
    bot.register_next_step_handler(message, redakcia)
    cur.close()
    conn.close()


def operacii(message):
    conn = sqlite3.connect("bd1.sql")
    cur = conn.cursor()
    conn.commit()
    global oper
    oper = message.text
    "UPDATE bd1 SET OPER = oper  WHERE id == chat.id"
    markup = types.InlineKeyboardMarkup(row_width=4)
    btn1 = types.InlineKeyboardButton("Возраст", callback_data="voz")
    btn2 = types.InlineKeyboardButton("Вес", callback_data="veS")
    btn3 = types.InlineKeyboardButton("Рост", callback_data="rost")
    btn4 = types.InlineKeyboardButton("Пол", callback_data="pl")
    btn5 = types.InlineKeyboardButton("Жалобы", callback_data="pois")
    btn6 = types.InlineKeyboardButton("Операции", callback_data="operation")
    btn7 = types.InlineKeyboardButton("Хронические заболеваний", callback_data="chronik")
    btn8 = types.InlineKeyboardButton("Назад", callback_data="nazad")
    markup.add(btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8)
    bot.send_message(message.chat.id,
                     f"Вот ваши данные: \nПол: {sex} \nВозраст: {age}\nРост: {height}\nВес: {ves} \nВаш специалист: {clf.predict(poison)[0].upper()} \nПеренесённые операции:{oper} \nХронические заболевания:{chron}",
                     reply_markup=markup)
    bot.send_message(message.chat.id, "Перенесенные операции успешно изменены")
    bot.register_next_step_handler(message, redakcia)
    cur.close()
    conn.close()
def karta(message):
    import overpass
    from geopy.geocoders import Nominatim
    import osmnx as ox
    import networkx as nx
    import os
    from selenium import webdriver
    import time
    bot.send_message(message.chat.id,"Работаем...")
    adress=message.text
    geolocator = Nominatim(user_agent="OOOOOOO")
    location = geolocator.geocode(adress)
    api = overpass.API()
    response_1 = \
        api.Get(f'node["amenity"="doctors"](around:10000,{location.latitude}, {location.longitude});out;')[
            'features']
    response_2 = \
        api.Get(f'node["amenity"="clinic"](around:10000,{location.latitude}, {location.longitude});out;')[
            'features']
    response_3 = \
        api.Get(f'node["amenity"="hospital"](around:10000,{location.latitude}, {location.longitude});out;')[
            'features']
    clinic, doctors = [], []
    for i in response_1:
        if 'name' in i["properties"]:
            a = i['geometry']['coordinates']
            doctors.append([a, i["properties"]['name']])
    for i in response_2:
        if 'name' in i["properties"]:
            a = i['geometry']['coordinates']
            doctors.append([a, i["properties"]['name']])

    for i in response_3:
        if 'name' in i["properties"]:
            a = i['geometry']['coordinates']
            doctors.append([a, i["properties"]["name"]])
    for i in doctors:
        if i not in clinic:
            clinic.append(i)

    # [37.840232, 51.295029];
    start_point = (51.298038, 37.833202)
    end_point = (51.295029, 37.840232)

    G = ox.graph_from_point(start_point, dist=5000, network_type='walk')

    orig_node = ox.nearest_nodes(G, start_point[1], start_point[0], False)

    dest_node = ox.nearest_nodes(G, end_point[1], end_point[0], False)

    route = nx.shortest_path(G,
                             orig_node,
                             dest_node,
                             weight='length')

    shortest_route_map = ox.plot_route_folium(G, route,
                                              tiles='openstreetmap')
    shortest_route_map.save('maaap.html')
    mapFname = 'maaap.html'
    mapUrl = 'file://{0}/{1}'.format(os.getcwd(), mapFname)
    driver = webdriver.Firefox()
    driver.get(mapUrl)
    # wait for 5 seconds for the maps and other assets to be loaded in the browser
    time.sleep(15)
    driver.save_screenshot('output.png')
    driver.quit()
    file = open("output.png", 'rb')
    bot.send_photo(message.chat.id, file)
    bot.register_next_step_handler(message, profile)


@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open('new_file.ogg', 'wb') as new_file:
        new_file.write(downloaded_file)
        new_file.close()
    data, samplerate = sf.read('new_file.ogg')
    sf.write('new_file.wav', data, samplerate)



def speech_to_text(file):
    r = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='ru-RU')
        return text
bot.infinity_polling()
