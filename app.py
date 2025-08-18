from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from collections import deque
import firebase_admin
import os
from firebase_admin import credentials, firestore
from datetime import datetime

cred = credentials.Certificate('/etc/secrets/firebasekey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(
    temperature=0, 
    groq_api_key=os.environ.get("GROQ_API_KEY"), 
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
)

leetcode_topic_tree = {
    'intervals': ['heaps/priority queues'],
    'greedy algorithms': ['heaps/priority queues'],
    'advanced graphs': ['heaps/priority queues', 'graphs'],
    'math & geometry': ['bit manipulation', 'graphs'],
    '2D dynamic programming': ['1D dynamic programming', 'graphs'],
    'bit manipulation': ['1D dynamic programming'],
    'heaps/priority queues': ['trees'],
    'graphs': ['backtracking'],
    '1D dynamic programming': ['backtracking'],
    'tries':['trees'],
    'backtracking':['trees'],
    'trees':['binary search', 'linked lists'],
    'binary search': ['two pointers'],
    'sliding window': ['two pointers'],
    'linked lists': ['two pointers'],
    'two pointers': ['arrays & hashing'],
    'stacks': ['arrays & hashing'],
    'arrays & hashing': []
}

def practice_dialog_flow(username, userinput):

    today = datetime.now().strftime('%d/%m/%Y')
    docs = db.collection('settings').where('userId', '==', username).stream()
    doclist = [doc.to_dict() for doc in docs]
    user_settings = doclist[0]
   
    lastpopped = user_settings.get('lastpopped', "")

    #theres no need to run below code if the last popped date is today
    #this means the user has already been given their flashcards for today
    if(lastpopped != today):
        questions = deque(user_settings.get('questions', []))

        for i in range(user_settings.get('learningrate', 1)):
            new_flashcard_id = questions.popleft()
            flashcard = {
                "problem":new_flashcard_id,
                "daysleft":0,
                "state":"ask question",
                "lastcompleted":"*",
                "increment":1,
                "userId":username
            }
            doc_ref = db.collection('flashcards').document()
            doc_ref.set(flashcard)

        # Get today's date in dd/mm/yyyy format
        today = datetime.now().strftime('%d/%m/%Y')
    
        # Update settings document with remaining questions and lastpopped
        settings_docs = db.collection('settings').where('userId', '==', username).get()
        for doc in settings_docs:
            doc.reference.update({
                'questions': list(questions),
                'lastpopped': today
            })

        #return str(user_settings)

    question = "" #these will be the strings to store one of the two strings
    evaluation = ""
    
    # Add the new flashcard evaluation logic here
    flashcards_ref = db.collection('flashcards')
    query = flashcards_ref.where('userId', '==', username).where('state', '==', 'evaluate response').limit(1)
    docs = query.get()

    if docs:
        flashcard_doc = docs[0]  # Get first matching document
        flashcard_data = flashcard_doc.to_dict()
        
        # Evaluate the response
        evaluation, solution = evaluateflashcardresponse(userinput, flashcard_data['problem'])

        result = evaluation[0].split(",")[0].strip()
        updates = {}

        print(result)

        if(result == "correct"):
            updates = {
                'daysleft': flashcard_data['increment'],
                'increment': flashcard_data['increment'] + 1,
                'state': 'ask question',
                'lastcompleted': today #do this provided the evaluation was successful
            }
        elif(result == "almost correct"):
            updates = {
                'daysleft': max(flashcard_data['increment'] - 1, 0),
                'increment': flashcard_data['increment'],
                'state': 'ask question',
                'lastcompleted': today #do this provided the evaluation was successful
            }
            evaluation.extend(solution)
        else:
            updates = {
                'state':'ask question'
            }
            evaluation.extend(solution)
        
        flashcard_doc.reference.update(updates)
    
    zero_days_query = flashcards_ref.where('userId', '==', username).where('daysleft', '==', 0).limit(1)
    zero_days_docs = zero_days_query.get()
        
    if zero_days_docs:
        flashcard_doc = zero_days_docs[0]
        flashcard_data = flashcard_doc.to_dict()

        question = createflashcardquestion(flashcard_data['problem'])
            
        flashcard_doc.reference.update({
            'state': 'evaluate response'
        })
    else:
        question = "REMOVE STATE"

    return [evaluation, question]

def practice_specific_topics_flow(userinput, memorylist):

    practice_specific_responses = []
    questionformatted = []
    evaluationformatted = []

    if(memorylist and memorylist[0][1] == 1):
       evaluationformated = evaluateflashcardresponse(userinput, memorylist[0][0])
       memorylist.popleft()

    if memorylist:
        questionformatted = createflashcardquestion(memorylist[0][0])

    practice_specific_responses.extend(questionformatted)
    practice_specific_responses.extend(evaluationformatted)

    return practice_specific_responses, memorylist


def evaluateflashcardresponse(userinput, problemid):
    # Dummy function - implement actual evaluation logic here
    # return "Your solution to problem " + str(problemid) + "has been evaluated!"
    leetcode_solutions_ref = db.collection('leetcode_solutions')
    query = leetcode_solutions_ref.where('id', '==', problemid).limit(1)
    docs = query.get()

    correctanswer = ""
    leetcodequestion = ""

    if docs:
        leetcode_solution_doc = docs[0]
        leetcode_solution_data = leetcode_solution_doc.to_dict()
        correctanswer = leetcode_solution_data.get("solution", "")
        leetcodequestion = leetcode_solution_data.get("intro", "")

    check_answer_chain = check_answer_prompt | llm 
    evaluation = check_answer_chain.invoke(input={'leetcode_question_data': leetcodequestion, 'userinput_data': userinput, 'correctanswer_data': correctanswer})
    
    evaluationformatted = [evaluation.content.strip()]
    solutionformatted = ["Here is the correct solution:"]
    solutionformatted.extend(correctanswer.split("\n"))
    #evaluationformatted.extend(solutionformatted)

    return evaluationformatted, solutionformatted


def createflashcardquestion(problemid):
    #return "Give an answer to problem " + str(problemid)
    leetcode_solutions_ref = db.collection('leetcode_solutions')
    query = leetcode_solutions_ref.where('id', '==', problemid).limit(1)
    docs = query.get()

    question = "question for " + str(problemid)

    if docs:
        leetcode_solution_doc = docs[0]
        leetcode_solution_data = leetcode_solution_doc.to_dict()
        question = leetcode_solution_data.get("intro", question)

    questionformatted = ['Solve the following Leetcode problem below (or tell me to stop early if you want):']
    question = question.split("\n")
    questionformatted.extend(question)

    return questionformatted


def get_topic_ancestors(topics):

    ancestors = []

    def traversal(topic, level):

        ancestors.append([level, topic])

        for child in leetcode_topic_tree[topic]:
            traversal(child, level + 1)

    for topic in topics:
        traversal(topic, 0)

    # Sort the ancestors by level in descending order
    ancestors.sort(key=lambda x: x[0], reverse=True)
    # Extract only the topic names from the sorted list
    ancestors = [topic for level, topic in ancestors]
    # Remove duplicates while maintaining order
    seen = set()
    ancestors = [x for x in ancestors if not (x in seen or seen.add(x))]
    
    return ancestors

def detect_early_stopping(userinput):
    # Dummy function - implement actual stop detection logic here
    # return True if user wants to stop, else False
    early_stopping_chain = early_stopping_prompt | llm 
    outputmessage = early_stopping_chain.invoke(input={'page_data': userinput})
    return (outputmessage.content.strip().lower() == 'yes')

state_classify_prompt = PromptTemplate.from_template(
        """
        ### USER MESSAGE FOR CLASSIFICATION:
        {page_data}
        ### INSTRUCTION:
        Classify the user message as a list of categories based on the following:
        specific: asks to practice specific topics and may specify the topic names as well
        practice: asks to practice their assigned problems in general, no topic names are mentioned
        view: asks to view their learning plan or topics in their problem set or similar phrases to that
        rate: asks to change their learning rate, may specify number of new problems to practice a day additionally or the need to change it
        struggle: says they are struggling or finding difficulty with specific topics, may name the topics after specifying they are struggling or finding difficulty with it 
        performance review: asks to give a performance review of their leetcode problems
        other: if the user message does not fit any of the above categories, classify it as other
        return as a single category or a list of categories separated by commas.
        if 'practice' and 'specific' are both identified, classify it as 'specific' only.
        order the categories based on how the user phrased their message if more than one category is identified.
        ### NO PREAMBLE OR PROAMBLE
        """
)

early_stopping_prompt = PromptTemplate.from_template(
    """
    ### USER MESSAGE FOR POTENTIAL EARLY STOPPING:
    {page_data}
    ### INSTRUCTION
    Determine if the user message clearly indicates they want to stop, quit, exit, or pause.
    Return 'yes' only if the intent is to end or pause the session (e.g. "I’m done for today", "let’s stop here", "quit now").
    Return 'no' if the word "stop" is used in any other context not related to ending or pausing the session (e.g., "stop making mistakes", "my code stops working").
    or if there is no clear indication of wanting to stop.
    If the user message does not clearly indicate a desire to stop, Return 'no'.
    ### NO PREAMBLE OR PROAMBLE
    """
)


learning_rate_prompt = PromptTemplate.from_template(
        """        
        ### USER MESSAGE ABOUT LEARNING RATE:
        {page_data}
        ### INSTRUCTION:
        the user message may specify a number of new
        problems to practice a day additionally or not or it might just a be a number on its own. If the user message does not specify a relevant number, return 'no number specified'.
        If the user message specifies a number, return the number mentioned as a single integer e.g. '5', '10','15' etc. and nothing else
        ### NO PREAMBLE OR PROAMBLE
        """
)

yes_no_classification_prompt = PromptTemplate.from_template(
        """
        ### USER MESSAGE FOR CLASSIFICATION:
        {page_data}
        ### INSTRUCTION:
        classify the response as either a 'yes' or 'no' type response and return the classification
        ### NO PREAMBLE OR PROAMBLE
        """
)

performance_prompt = PromptTemplate.from_template(
        """
        ### HASHMAP OF TOPIC AREAS AND THEIR SCORES:
        {page_data}
        ### INSTRUCTION:
        Based on the hashmap of topic areas and their scores, give a performance review of the user's leetcode problems.
        The performance review should be a short paragraph comparing the topic areas performance without mentioning the scores.
        give a ranking of the topic areas based on the scores, with the highest score being the best performing topic area without mentioning the scores.
        only if more than one topic area is present, a ranking as a numbered list with the topic area name should be given after a newline character.
        Refer to the user as 'you' in the performance review.
        ### NO PREAMBLE OR PROAMBLE
        """
)

check_answer_prompt = PromptTemplate.from_template(
        """
        ### LEETCODE QUESTION:
        {leetcode_question_data}
        ### USER ANSWER FOR THE LEETCODE QUESTION:
        {userinput_data}
        ### CORRECT ANSWER FOR THE LEETCODE QUESTION:
        {correctanswer_data}
        ### INSTRUCTION:
        By comparing the user's leetcode solution with the mentioned time and space complexity with the
        correct answer along with it's time and space complexity, classify how accurate the user's response is into the
        following:

        correct: User answer matches the correct answer in terms of accuracy and efficiency and they've mentioned the correct time and space complexity
        almost correct: User answer works the same as the correct answer but is not as efficient or the user has mentioned a correct and efficient solution but mentioned the wrong time or space complexity
        wrong: User answer is incorrect as it doesn't perform the same action as the correct answer

        variable, function and class names obviously don't need to be the same when comparing the user answer and correct answer.
        as well as classifying, give a short explanation for the classification reason.
        return a response in the form "[classification],[reason]"
        ### NO PREAMBLE OR PROAMBLE
        """
)

specific_topic_prompt = PromptTemplate.from_template(
        """
        ### USER MESSAGE FOR STRUGGLE:
        {page_data}
        ### INSTRUCTION:

        here are all the leetcode topic areas:
        arrays & hashing
        two pointers
        stacks
        binary search
        sliding window
        linked lists
        tries
        trees
        backtracking
        heaps/priority queues
        graphs
        1D dynamic programming
        2D dynamic programming
        bit manipulation
        greedy algorithms
        intervals
        advanced graphs
        math & geometry

        Identify the leetcode topic areas from the list above mentioned in the user message, return as a single category or a list of categories separated by commas.
        If no topic areas are mentioned return 'no topic areas mentioned'. The list items must be the exact item names from the list above.
        ### NO PREAMBLE OR PROAMBLE
        """
)

class AgentRequest(BaseModel):
    userinput: str
    username: str
    userstate: list
    userstatecount: int
    usershorttermmemory: dict

class AgentResponse(BaseModel):
    agentoutput: list
    newstate: list
    newstatecount: int
    newshorttermmemory: dict

@app.post("/api/agentresponse", response_model=AgentResponse)
async def handle_post(data: AgentRequest):
    userinput = data.userinput
    username = data.username
    userstate = data.userstate
    userstatecount = data.userstatecount
    usershorttermmemory = data.usershorttermmemory

    if not username:
        raise HTTPException(status_code=400, detail="No username provided")
    if not userinput:
        raise HTTPException(status_code=400, detail="No user input provided")
    if not userstate:
        raise HTTPException(status_code=400, detail="No user state provided")
    
    print(username)
    print(userinput)
    print(userstate)
    print(userstatecount)
    print(usershorttermmemory)
    
    statequeue = deque()
    if(userstate[0] == 'classify'):
        state_classify_chain = state_classify_prompt | llm 
        outputmessage = state_classify_chain.invoke(input={'page_data': userinput})
        states = outputmessage.content.strip().split(',')
        states = set([state.strip() for state in states]) # Clean up any extra spaces
        statequeue = deque(states)
        statequeue.append('classify') # This is the default state to classify the user input
    else:
        statequeue = deque(userstate)

    responses = []
    responsefinished = False

    while not responsefinished:

        if(statequeue[0] == 'practice'):
            if(userstatecount == 2 and detect_early_stopping(userinput)):
                    responses.append("Okay! Stopping the practice session now. progress is automatically saved so you can continue again later.")
                    statequeue.popleft()
                    userstatecount = 1
            else:
                practicestateresponses = practice_dialog_flow(username, userinput)
                responses.extend(practicestateresponses[0])
                if('REMOVE STATE' in practicestateresponses):
                    statequeue.popleft()
                    userstatecount = 1
                else:
                    responses.extend(practicestateresponses[1])
                    responsefinished = True
                    userstatecount = 2
        elif(statequeue[0] == 'specific'):
            #responses.append("practice specific topics")
            if(userstatecount == 1):
                specific_topic_chain = specific_topic_prompt | llm
                outputmessage = specific_topic_chain.invoke(input={'page_data': userinput})

                if(outputmessage.content.strip() != 'no topic areas mentioned'):
                    usershorttermmemory['specific'] = []
                    topics = outputmessage.content.strip().split(',')
                    topics = set([topic.strip() for topic in topics])

                    for topic in topics:
                        topicquestions = []
                        query = db.collection('leetcode_solutions').where('topic', '==', topic).stream()
                        for doc in query:
                            solution_data = doc.to_dict()
                            topicquestions.append([solution_data['id'], solution_data['order']])
                        topicquestions.sort(key=lambda x: x[1])
                        usershorttermmemory['specific'].extend([[question[0], 0] for question in topicquestions])
                else:
                    responses.append("Could you mention the leetcode topic areas you want to practice?")
                    responsefinished = True
            
            if(not responsefinished):
                if(userstatecount == 2 and detect_early_stopping(userinput)):
                    responses.append("Okay! Stopping the practice session now. progress is automatically saved so you can continue again later.")
                    statequeue.popleft()
                    userstatecount = 1
                else:
                    memorylist = deque(usershorttermmemory.get('specific', []))

                    if(memorylist[0][1] == 1):
                        evaluation, solution = evaluateflashcardresponse(userinput, memorylist[0][0])
                        result = evaluation[0].split(",")[0].strip()
                        if(result != "correct"):
                            memorylist.append([memorylist[0][0], 0])  # Re-add the question with incremented state
                            evaluation.extend(solution)
                        responses.extend(evaluation)
                        memorylist.popleft()
                    if memorylist:
                        responses.extend(createflashcardquestion(memorylist[0][0]))
                        memorylist[0][1] = 1 # Set the state to 1 to evaluate the response later
                        responsefinished = True
                        userstatecount = 2
                    else:
                        statequeue.popleft()
                        userstatecount = 1

                    usershorttermmemory['specific'] = list(memorylist)  # Update the short term memory with the remaining questions 
        elif(statequeue[0] == 'struggle'):

            if(userstatecount == 1):
                state_struggle_chain = specific_topic_prompt | llm 
                outputmessage = state_struggle_chain.invoke(input={'page_data': userinput})

                if(outputmessage.content.strip() != 'no topic areas mentioned'):
                    topics = outputmessage.content.strip().split(',')
                    topics = set([topic.strip() for topic in topics])
                    if topics:
                        topic_ancestors = get_topic_ancestors(topics)
                        usershorttermmemory['struggle'] = topic_ancestors

                        topicmention = 'Okay!, the area for you to work on is ' + topic_ancestors[0]
                        if(len(topic_ancestors) > 1):
                            topicmention = 'Okay!, the area for you to work on are' + ', '.join(topic_ancestors[:-1]) + " and " + topic_ancestors[-1]
                        responses.append(topicmention)
                        responses.append('Would you like me to make changes to your learning plan?')
                        #statequeue.popleft()
                        responsefinished = True
                        userstatecount += 1
                else:
                    responses.append("Could you mention the leetcode topic areas you are struggling with? I will add the relevant topics to your learning plan! ")
                    responsefinished = True
            else:
                #classify yes/no from user input
                yes_no_chain = yes_no_classification_prompt | llm 
                outputmessage = yes_no_chain.invoke(input={'page_data': userinput})

                if(outputmessage.content.strip() == "yes"):
                    topics = usershorttermmemory.get("struggle", [])
                    questions = []

                    for topic in topics:
                        topicquestions = []
                        query = db.collection('leetcode_solutions').where('topic', '==', topic).stream()
                        for doc in query:
                            solution_data = doc.to_dict()
                            topicquestions.append([solution_data['id'], solution_data['order']])
                        topicquestions.sort(key=lambda x: x[1])
                        questions.extend([question[0] for question in topicquestions])
                    
                    settings_docs = db.collection('settings').where('userId', '==', username).get()
                    for doc in settings_docs:
                        doc.reference.update({
                            'questions': questions
                        })

                    topicmention = topics[0]
                    if(len(topics) > 1):
                        topicmention = ', '.join(topics[:-1]) + " and " + topics[-1]
                    responses.append("Okay!, i've added " + topicmention + " to your practice set")
                else:
                    responses.append("No worries!")
                statequeue.popleft()
                userstatecount = 1

        elif(statequeue[0] == 'performance review'):
            #responses.append("give a performance review of your leetcode problems. ")
            #statequeue.popleft()
            #userstatecount = 1
            topicmap = {}
            query = db.collection('flashcards').where('userId', '==', username).stream()
            for doc in query:
                flashcard_data = doc.to_dict()
                nestedquery = db.collection('leetcode_solutions').where('id', '==', flashcard_data['problem']).stream()
                for nesteddoc in nestedquery:
                    leetcode_solution_data = nesteddoc.to_dict()
                    topicmap[leetcode_solution_data['topic']] = topicmap.get(leetcode_solution_data['topic'], 0) + flashcard_data['increment']

            performance_chain = performance_prompt | llm 
            outputmessage = performance_chain.invoke(input={'page_data': str(topicmap)})
            responses.append("Here is your performance review: ")
            responses.append(outputmessage.content.strip())
            statequeue.popleft()
            userstatecount = 1
        elif(statequeue[0] == 'view'):
            #responses.append("view your learning plan or topics in your problem set or similar phrases to that. ")
            #statequeue.popleft()
            #userstatecount = 1
            topics = []
            questions = []

            settings_docs = db.collection('settings').where('userId', '==', username).get()
            for doc in settings_docs:
                user_settings = doc.to_dict()
                questions = user_settings.get('questions', [])

            for id in questions:
                query = db.collection('leetcode_solutions').where('id', '==', id).stream()
                for doc in query:
                    leetcode_solution_data = doc.to_dict()
                    if leetcode_solution_data['topic'] not in topics:
                        topics.append(leetcode_solution_data['topic'])

            responses.append("Here is your learning plan: ")
            if topics:
                responses.append(", ".join(topics))

            statequeue.popleft()
            userstatecount = 1
        elif(statequeue[0] == 'rate'):
            #responses.append("change your learning rate, may specify number of new problems to practice a day additionally. ")
            #statequeue.popleft()
            #userstatecount = 1

            settings_docs = db.collection('settings').where('userId', '==', username).get()
            learningrate = settings_docs[0].to_dict().get('learningrate', 1)
            
            learning_rate_chain = learning_rate_prompt | llm 
            outputmessage = learning_rate_chain.invoke(input={'page_data': userinput})
            print(outputmessage.content.strip())
            userinput = outputmessage.content.strip()
            if(userinput == 'no number specified'):
                responses.append(f"Your current learning rate is {learningrate} problems per day. Would you like to change it?")
                responses.append("If so, Please specify a number of problems to practice a day for the learning rate.")
                responsefinished = True
            else:
                try:
                    new_learning_rate = int(userinput)
                    for doc in settings_docs:
                        doc.reference.update({
                            'learningrate': new_learning_rate
                        })
                    responses.append(f"Okay! I've set your learning rate to {new_learning_rate} problems per day.")
                    statequeue.popleft()
                    userstatecount = 1
                except ValueError:
                    responses.append("Please specify some number for the learning rate.")
                    responsefinished = True
        else:
            responses.append("Sorry, I don't think I can help with that. ")
            statequeue.popleft()
            userstatecount = 1

        # the condition below will occur if the the previous states were popped off the queue implying we are in the classify state
        if(statequeue[0] == 'classify'):
            responses.append('If required, feel free to ask me more leetcode related questions')
            responsefinished = True

    response = AgentResponse(agentoutput=responses, newstate=list(statequeue), newstatecount=userstatecount, newshorttermmemory=usershorttermmemory)
    return response
