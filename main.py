import nltk
import numpy as np
import random
import string

if __name__ == '__main__':

    do_basic_loop = False
    if do_basic_loop:
        response = ''
        while response not in {'quit'}:
            response = input('Yes? ')
            print(response)

    input_file = './data/chatbot.txt'
    errors_ = 'ignore'
    with open(input_file, 'r', errors=errors_) as input_fp:
        raw_text = input_fp.read()
