from wikipediaapi import Wikipedia

if __name__ == '__main__':
    client = Wikipedia('en')
    page = client.page('Chatbot')
