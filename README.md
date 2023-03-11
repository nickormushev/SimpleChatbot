# SimpleChatbot

## Description
I created this chatbot with the idea of it responding to a few predifined questions. It should work a bit like an FAQ section. When a person has a question it matches it to a class and choses a response based on that. 

## How to use
The data for the questions it can answer is provided in the data.json file. To add more responses you have to add a new class with a few example patterns to match for that class. If the class matches a random response is chosen from the responses field. To run the program just install the dependencies and run the main_discord.py file with python. For discord an environemnt variable named BOT_TOKEN needs to be exported with the value of a bot token connected to a registered discord bot and the 'discord' argument needs to be passed when running the program.
