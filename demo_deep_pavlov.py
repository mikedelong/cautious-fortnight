from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill

if __name__ == '__main__':
    bye = PatternMatchingSkill(['Goodbye world!', 'See you around'], patterns=['bye', 'ciao', 'see you'])
    hello = PatternMatchingSkill(responses=['Hello world!'], patterns=['good day', 'hello', 'hi'])
    fallback = PatternMatchingSkill(['I\'m sorry; I don\'t understand.'])
    agent = DefaultAgent([hello, bye, fallback], skills_selector=HighestConfidenceSelector())
    for query in ['Hello', 'Bye', 'Or not']:
        response = agent([query])
        print('Q: {} A: {}'.format(query, response[0]))
