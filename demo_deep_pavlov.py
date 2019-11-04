# from deeppavlov.core.agent import Agent
# from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill

bye = PatternMatchingSkill(['Goodbye world!', 'See you around'], patterns=["bye", "ciao", "see you"])
hello = PatternMatchingSkill(responses=['Hello world!'], patterns=["hi", "hello", "good day"])
fallback = PatternMatchingSkill(["I don't understand, sorry"])
