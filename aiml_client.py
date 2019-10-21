# https://www.devdungeon.com/content/ai-chat-bot-python-aiml

import aiml

# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.learn("std_startup.xml")
kernel.respond("load aiml b")

# Press CTRL-C to break this loop
while True:
    print(kernel.respond(input("Enter your message >> ")))
