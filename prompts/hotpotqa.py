#------------------------
# Prompt for Generation
#------------------------

cot_prompt = '''
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
{examples}
Question: {question}
'''


propose_prompt = '''
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Here are some examples.
{examples}

Important Rules:
1. Each Thought and Action pair must be self-contained and logically independent from others. They should not rely on information that would be obtained in another Thought or Action.
2. When proposing Thoughts, consider multiple plausible directions or strategies for reasoning, but avoid assuming information that would depend on the outcome of another Thought or Action.
3. If an Action references prior knowledge, ensure that it is explicitly stated in the Current context or in the Question.

Now, given the current context of the question and the reasoning, provide 4 possible future reasoning paths.
For each path, provide a Thought and an associated Action.

Example Structure:
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Current context:

Thought 1: I need to find out the professions of Nicholas Ray.
Action 1: Search[Nicholas Ray]  
Observation 1: Nicholas Ray was an American film director, screenwriter, and actor, best known for "Rebel Without a Cause".

Thought 2: I need to find out the professions of Elia Kazan to compare with Nicholas Ray's professions.
Action 2: Search[Elia Kazan]  
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter, and actor.

4 Possible Thought,Action pairs (each self-contained and independent of others):
Thought: Based on the shared professions of director, screenwriter, and actor, I can conclude that these professions are indeed what Nicholas Ray and Elia Kazan have in common.
Action: Finish[director, screenwriter, actor]

Thought: I need to confirm whether Elia Kazan also worked as an actor, as this is mentioned in his search result.
Action: Lookup[actor]

Thought: I should verify if Nicholas Ray worked as a theatre director, which could be another shared profession.
Action: Lookup[theatre director]

Thought: I need to search if both Nicholas Ray and Elia Kazan are categorized under screenwriters on Wikipedia.
Action: Search[Nicholas Ray, Elia Kazan screenwriter]


Question: {question}
Current context: 
{context}
4 Possible Thought,Action Pairs (each self-contained and independent of others):
'''



#------------------------
# Prompt for Evaluation
#------------------------

value_prompt = '''
Evaluate the likelihood that the following reasoning path will lead to the correct answer (high/medium/low). 
Provide your evaluation based on the trajectory of the reaosning path as well as past information gathered, and judge whether this will lead to a correct answer. 
Provide a brief justification for this evaluation. You should put the evaluation (high/medium/low) at the very start of your evaluation

Question: 
What profession does Nicholas Ray and Elia Kazan have in common?

Context: 

Thought: I need to find out the professions of Nicholas Ray.
Action: Search[Nicholas Ray]  
Observation: Nicholas Ray was an American film director, screenwriter, and actor, best known for "Rebel Without a Cause".

Thought: Now I need to find out the professions of Elia Kazan to see if they share any.
Action: Search[Elia Kazan]  
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter, and actor.

Reasoning Path:
Thought: Both Nicholas Ray and Elia Kazan worked as directors and screenwriters, so I need to check if they also share the actor role.
Action: Lookup[actor]

Likelihood and Evaluation: (First word need to be either high, medium, or low)
High
This reasoning path has a high likelihood of leading to the correct answer. Both individuals' professions have been partly identified, and the user is now verifying if they share the actor role, which is likely to conclude their shared professions.

Question:
{question}

Context:
{reasoning_path}


Likelihood and Evaluation: (First word need to be either high, medium, or low)
'''



#-----------------------------------
# Prompt for example
#-----------------------------------

react_example_prompt = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup[eastern sector]
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: Search[High Plains]
Observation: High Plains refers to one of two distinct land regions:
Thought: I need to instead search High Plains (United States).
Action: Search[High Plains (United States)]
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]


Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action: Search[Milhouse]
Observation: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action: Lookup[named after]
Observation: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[Richard Nixon]


Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action: Search[Adam Clayton Powell]
Observation: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought: To find the documentary, I can search Adam Clayton Powell (film).
Action: Search[Adam Clayton Powell (film)]
Observation: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action: Finish[The Saimaa Gesture]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action: Search[Nicholas Ray]
Observation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action: Search[Elia Kazan]
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]


Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action: Search[Arthur's Magazine]
Observation: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action: Search[First for Women]
Observation: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: Finish[Arthur's Magazine]


Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action: Search[Pavel Urysohn]
Observation: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action: Search[Leonid Levin]
Observation: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action: Finish[yes]
"""
















