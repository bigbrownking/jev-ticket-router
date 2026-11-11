from dotenv import load_dotenv
load_dotenv() 

from jevbench.jev import make_jev
from jevbench.policy import decide

jev = make_jev("typesafe")
text = input("Тикет: ")
answers, usage, model = jev.ask(text)

for name, a in answers.items():
    print(name, {k: v for k, v in a.items() if k != "legend"})
print("\nРешение:", decide(answers))