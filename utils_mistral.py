import pandas as pd
from openai import OpenAI
import os

exemples_inputs = ["""Type : Open to change
<<< Context : Therapist: Okay, and what kinds of hobbies do you have?  Patient: Um, I like to read. And I like to go outside and like take walks and like hike and stuff. I'm very outdoorsy.  Therapist: So. Oh, cool. What are your plans for after high school?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type : Resistant to change
<<< Context : Therapist: Can you tell me what country we are in Patient: America. Therapist: What County are we in?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: hesitant begining of the dialogue
<<< Context : Therapist: guide yourself spending a lot of money on? Patient: Yeah Therapist: how much do you spend in a week or what your, your habit like right now?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change
<<< Context : Therapist: do I describe that then? dealing with my dad. Okay. Dealing with Dad.  Is there anything else you'd like to put on the agenda? Patient: I've been really tired lately just been sleeping a lot. Okay, last week or two. Therapist: So I put down tired and sleep. So we have dealing with dad and tired and sleep. Which thing do you want to talk about first?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Resistant to change
<<< Context: Therapist: How long have they been wanting to do this to you? Patient: all my life. I've done this all my life. They they're always listening. They're always around. They're always there. Therapist: always there?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Hesitant end of the dialogue
<<< Context : Therapist: Why don't you start reinsurance? Everything's going right. Okay, so that seems reasonable to you. Right? I mean, I, I guess Patient: but at the same time, like when I'm out doing my own thing, you know, I kind of want that privacy in that, you know, just being in the moment you know, because I know when I get back home, from back with you and I'm, nothing's changed because I've always been with you like we're still married. But when I'm out with my friends, and you know, I'm getting a couple phone calls during you know, the night over the course of the night and it's having to deal with my wife and having to make sure that she knows I'm okay and everything like that it it gets a little frustrating. It's a little, I mean, frustrating Therapist: Would that be a good compromise? It was a text or two. I guess. That's better than a phone call, right? Yeah. So not ideal, but you can see how it might decrease your anxiety a bit. Right? And maybe work toward a little bit more freedom. So that's how I can experiment you both combined to try.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change
<<< Context : Therapist: What do you have in mind? Patient: I just keep thinking about you know, losing this weight and I just really want to do it. I just I just can't get there and I don't know why. too frustrated. Its ridiculous Therapist: frustated Trying to lose the weight. Now you'd said before that you had a specific number of pounds of specific amount of weight to lose. You still have that in mind?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Resistant to change
<<< Context : Therapist: I mean, I'm feeling just lumped in with everybody else right now.  Patient: Well, you are because you're all the same. Of course, you're lumped in with everybody. I'm not gonna single you out. Not gonna make you you know, single you out because you're not doing anything. You don't do anything. There's all this abstract metaphysical stuff and there's no concrete. This is the way to fix it. This is the goal. There's nothing so what's the point? Therapist: I mean, it's there's something specific right now that I could say or do that that would fill, give you what you're wanting or need.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type : Resistant to change
<<< Context : Therapist: about bottle of vodka. And how long have you been drinking that much for? Patient: Well, as I say it's crept up on for Few years now, Therapist: a few years, can you make it a more specific two years? Three years? Five years?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""<<< Context : Therapist: alcohol and marijuana. How recently was this DUI? About a week ago? Okay, so what's been going on when that happened? Patient: Um, well, I am going to have to go to courtsoon. And I know that I'm probably gonna have to go I'd like treatment or something like that, but I really just don't want to. So I just, I'm just coming here because I don't know maybe I can like get out of it if I'm gettingsome type of treatment. But yeah, Therapist: get out of it. You mean get out of the charge?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""<<< Context : Patient: I want to completely cut out fast food. I want to have more set meals and I just want to eat healthy food. Okay. Patient: I think I at least needs to go to the grocery. Okay.  Therapist: Is that something that you haven't been doing?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""<<< Context : Therapist: just your vehicle have stuff in it now. There you go. Patient: Yeah. So I actually picked something up that I saw. That was for sale on the side of the road on the way here was a little bedside table. So it was only a few bucks. Therapist: Right. So you have some things you buy, you have some things that you otherwise obtain that are in the house. They're already there. And you have trouble parting with them. ever tried to throw anything out or do anything away? 
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""<<< Context : Therapist: How would you rate yourself when you're in bed before you go to work?  Patient: Probably like an eight. Therapist: So an eight sometimes nine, right now five, so still not feeling great. Have you had to pick an emotion of standing out above the rest in anxiety, sadness, sense of dread. What would be the worst right now?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type Hesitant beginning of the dialogue
<<< Context : Therapist: Do you have a plan to call Frank? Like when he's out? Or is it something that you would do if you just felt the need to or Therapist: Were you calling for some specific reason? Therapist: Why don't you start reinsurance? Everything's going right. Okay, so that seems reasonable to you. Right? I mean, I, I guess
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Resistant to change
<<< Context : Therapist: Okay. Did you have any other questions about anything that you read on that shape there? Patient: Um, no. it all seems fine with me. Yeah.  Therapist: Yeah. Okay. So if you wouldn't mind just popping your signature at the bottom there. Thanks.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change
<<< Context : Therapist: In half an hour. So notice you're looking around a little bit more now. helps a bit more with that pillow doesn't it? That was a good idea for you to put the pillow there. So how long do you think you'd like to be here? If you had a choice of how long you were going to be here with me in this conversation, how long would it be?  Patient: 15 minutes, maybe another 15 minutes.  Therapist: Okay, maybe, maybe
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change

<<< Context : Therapist: And today you're able to kind of shove that internal critic back a little. And allow that internal critic to see how strongly you feel with the decision you've made. Patient: Yeah, that I very much Therapist: do you feel is helpful technique?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""<<< Context : Therapist: So it doesn't matter that there's more than one gallon going in, is because it's the same product.  Patient: Exactly. Yes. Therapist: So I could see now this, this would be more of a daily thing. You said, Right. Yeah. Because at first I was thinking daily. That's a lot of individual shopping trips, but it applies to all transaction.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Resistant to change
<<< Context : Therapist: So you have the use it takes away pain for you to take swings xiety feels good. But then you have these consequences of the law. And you figure those two things out. Patient: I don't know. I don't know if there is what was your wealth of knowledge. So like not as just Therapist: You can imagine that. So it was a good the other side. Can you imagine having further it says law enforcement like more charges in the future.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change
<<< Context : Therapist: Yeah, that's sure. Patient: Yeah. and not have to stress right at the end. Therapist: Yeah. And not waste that time in the middle. Yeah. Right. That's, that's really eating more time and your show takes
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Hesitant end of the dialogue
<<< Context : Therapist: It's kind of a social thing for you not like a major priority. Yeah. How much do you think you're drinking? Patient: I guess I would say like every other every other weekend, maybe every every two weeks. On occasion, maybe There's a basketball game going on or a football game I'll have a beer just to drink with my boys. But other than that, I don't really see it as much of a of a problem. Therapist: So you're drinking maybe half the time depending on what's going on that weekend.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change

<<< Context : Therapist: Hello stone,  Patient: yes. Therapist: Hi, I'm PJ Daniels. Sorry, let me wash my hands.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Hesitant end of the dialogue
<<< Context : Therapist: So I think I sort of have all the information I need to make sure that your wishes are respected. Just one last time. Anything else you want to talk about today? Patient: Oh, you answered my question. I was so worried about this tube.  Therapist: Okay. All right. Well, it's very nice to meet you and Stone.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Resistant to change
<<< Context : Therapist: Hello Kathy, how are you today?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change
<<< Context : Therapist: Dover County? What state Are we in? Patient: Delaware. Therapist: So, I'm going to test your memory. I want to read these three words. And I want you to repeat them back to me.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Resistant to change
<<< Context : Therapist: So you're looking down so you take these quizzes in class? Patient: Yes. Therapist: like a paper pencil ?
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Hesitant end of the dialogue
<<< Context : Therapist: And you evaluate that they weren't a failure. Patient: Right. Therapist: But If it was  yourself. You're being a little more harsh. 
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Hesitant beginning of the dialogue
<<< Context : Patient: it is good, how are you? Therapist: I'm doing well. Thanks for asking. Therapist: What's been going on this last week? 
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Hesitant end of the dialogue
<<< Context : Therapist: So you're realizing to how much it really takes to focus attention on on the baby. And to be able to Do it and it's like I it's hard to do both. It's hard to really think about doing both. Patient: I need his help with that. Yeah, I don't need to be worrying about him too. Therapist: Hmm
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Resistant to change
<<< Context : Therapist: But first I want to get through kind of a piece on confidentiality. Patient: Okay. So like attorneys what you what you can do and not do. Therapist: We kind of I'm glad you brought that up, about attorneys. Attorneys and clients have confidentiality, and counselors and clients have confidentiality, but the two types are different. So I'll explain to you how it worked between you and I. But no, that is a little bit different than it works between the attorney and their client.
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Open to change
<<< Context : Therapist: That makes sense? Patient: That makes a lot of sense. Therapist: Okay. The other exceptions are insurance company. So when you fill out your screen packet insurance Insurance companies have access to like progress knows the assessment, anything on the chart and they're generally responsible this information I don't know of any instances where they're irresponsible with it. But I just think it's important that you know that anything I write in the chart anything in the chart itself technically the insurance company can ask for and periodically could review. Again, I've never seen in my career any difficulty with that, but as far as I know that information would be transmitted off site so to speak. And when this agency bills it would be transmitted your information be transmitted off site to the insurance company 
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>""","""Type: Hesitant end of the dialogue
<<< Context : Therapist: Hmm Patient: What does it mean light reading. Therapist: This sounds hard
            Generate the patient's next utterance and its category in the form {"Category": [category]
"Patient's Utterance" : [utterance]}  >>>"""]

exemples_outputs = ["""{"Category" : Sharing personal information or Describe past event
"Patient's Utterance" : Well, I plan to apply to college and maybe you know, have a good career and something like that. Hopefully.}""","""{"Category": "Sharing personal information or Describe past event",
"Patient's Utterance" : "We are in Kent County, Kent County."}""","""{"Category": "Sharing personal information or Describe past event",
"Patient's Utterance" :"about 200$"}""","""{"Category": "Sharing personal information or Describe past event",
"Patient's Utterance" : "Definitely my dad."}""","""{"Category": "Sharing personal information or Describe past event",
"Patient's Utterance" : "always there. Like, for example, they're in the lights right now. They're listening to everything that we're talking about. So that's the reason why I really can't talk about what happened on the bus because they're just going to use that against and then it's locked me up again. I think we take more minutes."}""","""{"Category": "Changing unhealthy behavior in the future",
"Patient's Utterance" : "Okay, so we could try and try it out."}""","""{"Category": "Changing unhealthy behavior in the future",
"Patient's Utterance" : "I would like 20 pounds 20 pounds. I mean, I would like more but I, I've tried to do more in the past and I don't even get past couple of days. So I can do 20 that would be great."}""","""{"Category": "Changing unhealthy behavior in the future",
"Patient's Utterance" : "How do I, how do you how do I go from point A to point B? How do I get better? How do I fix what's going on? I mean, that's the reason I came here. Right? I thought there was a reason I came here is to fix things. make things better."}""","""{"Category": "Sustaining unhealthy behavior in the future",
"Patient's Utterance" : "Well, I mean, as I say, it's crept up. I mean, initially, I just used to have a couple of drinks after work, right? Because I got a pretty responsible job. And I just found the drink calmed me down, you know. And then I got more responsibility, and I found a bit more alcohol helped me so. Yeah, yeah. So drinking about it"}""","""{"Category": "Sustaining unhealthy behavior in the future",
"Patient's Utterance" : "Yeah. I mean, I feel like they're probably gonna make me like go to like rehab or something like that. And I don't want to do that."}""","""{"Category": "Sustaining unhealthy behavior in the future",
"Patient's Utterance" : "No, not lately. I've just been grabbing food at work, or going through fast food on the way home. Okay."}""","""{"Category": "Sustaining unhealthy behavior in the future",
"Patient's Utterance" : "people have suggested that I do that, but I just can't bring myself to that. I mean, like my sister came over, she tried to get rid of something for me. couldn't do it. It's too hard. Yeah. Just think I might need it, you know, but if I need it, if I need to look back and reference that, that newspaper that she tried to throw away, we'll have it."}""","""
{"Category": "Sharing negative feeling or emotion",
"Patient's Utterance" : "Right now probably anxiety."}""","""{"Category": "Sharing negative feeling or emotion",
"Patient's Utterance" : "but at the same time, like when I'm out doing my own thing, you know, I kind of want that privacy in that, you know, just being in the moment you know, because I know when I get back home, from back with you and I'm, nothing's changed because I've always been with you like we're still married. But when I'm out with my friends, and you know, I'm getting a couple phone calls during you know, the night over the course of the night and it's having to deal with my wife and having to make sure that she knows I'm okay and everything like that it it gets a little frustrating. It's a little, I mean, frustrating"}""","""{"Category": "Sharing negative feeling or emotion",
"Patient's Utterance" : "I haven't, I'm feeling a little nervous."}""","""{"Category": "Sharing positive feeling or emotion",
"Patient's Utterance" : "Yeah, I feel better."}""","""{"Category": "Understanding or New Perspective",
"Patient's Utterance" : "Yeah, I do. I, you know, I think that separating the two sides helped me realize which one felt stronger."}""","""
{"Category": "Understanding or New Perspective",
"Patient's Utterance" : "Exactly. Yeah. I did really realize that that way. But it does. Yeah. Yeah."}""","""{"Category": "Understanding or New Perspective",
"Patient's Utterance" : "I guess I hadn't really considered that."}""","""{"Category": "Understanding or New Perspective",
"Patient's Utterance" : "I see where you're gonna. Absolutely."}""","""{"Category": "Understanding or New Perspective",
"Patient's Utterance" : "Yeah, I guess you can say that."}""","""{"Category": "Greeting or Closing ",
"Patient's Utterance" : "Call me, Dorothy."}""","""{"Category": "Greeting or Closing ",
"Patient's Utterance" : "Thank you."}""","""{"Category": "Greeting or Closing ",
"Patient's Utterance" : "Okay."}""","""{"Category": "Greeting or Closing ",
"Patient's Utterance" : "Okay. All right."}""","""{"Category": "Backchannel ",
"Patient's Utterance" : "Hmm hmm"}""","""{"Category": "Backchannel",
"Patient's Utterance" : "Yeah."}""","""{"Category": "Backchannel ",
"Patient's Utterance" : "Um"}""","""{"Category": "Asking for Medical Information ",
"Patient's Utterance" : "I don't understand what safety planning means."}""","""{"Category": "Asking for Medical Information ",
"Patient's Utterance" : "Okay. Now explain the differences to me."}""","""{"Category": "Asking for Medical Information ",
"Patient's Utterance" : "[WRONG] Can you explain light reading to me ?"}""","""{"Category": "Asking for Medical Information ",
"Patient's Utterance" : "[WRONG] Can you explain light reading to me ?"}"""]

def read_prompt_csv(role):
  if role == 'client':
      filename = 'DialogueEnvs/Users/prompts/client_prompts.csv'
  elif role == 'therapist':
      filename = 'DialogueEnvs/Users/prompts/therapist_prompts.csv'
  df = pd.read_csv(filename)
  intent_detail_list = []
  for index, row in df.iterrows():
      print(row)
      positive_examples = [row['positive example 1'],row['positive example 2'], row['positive example 3'], row['positive example 4'], row['positive example 5']]

      intent_detail_list.append({'intent': row['intent'].strip(),'definition': row['definition'],'positive_examples': positive_examples})
  return intent_detail_list


def get_completion_from_messages_local(messages, temperature=0.7):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


    response = client.chat.completions.create(
                                            model="local-model",
                                            messages=messages,
                                            temperature=temperature,
                                        )

    return response.choices[0].message.content


def create_message_client_generation_conditionned_da(intent_detail_list, intent, context,theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:
            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    themes = "Your are here to talk about"
    if theme == "Smoking":
        themes += "your smoking habbits."
    elif theme == "Drinking":
        themes += "your drinking habbits"
    else:
        themes += "your exercising habbits"
    system_prompt_template = theme+f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance and respect the intent given \
                                    from one of the following predefined categories:\n {intent_definition}\
                                     
                                            ####
                                              Here are some examples:\n {exemples}
                                            ####
                                     The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.\ 
                                     You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition.\
                                     The response must be short, no more than 2 or 3 sentences.\
                                     Use [laughter] to signal a patient laughter, [laughs] to signal a patient laugh, [sighs] to signal a patient sigh, [gasps] to signal a patient gasps, [clears throat] to signal a patient clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word.\
                                     
                                            <<<
                                              Context : {context}
                                              Generate the patient's next utterance with the intent: {intent}
                                            >>>
                                            Patient's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return messages


def create_message_client_generation_conditionned_da_vllm(intent_detail_list, intent, context, theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:
            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    themes = "Your are here to talk about"
    if theme == "Smoking":
        themes += "your smoking habbits."
    elif theme == "Drinking":
        themes += "your drinking habbits"
    else:
        themes += "your exercising habbits"
    system_prompt_template = theme + f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance and respect the intent given \
                                    from one of the following predefined categories:\n {intent_definition}\

                                           
                                     The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.\ 
                                     You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition.\
                                     The response must be short, no more than 2 or 3 sentences.\
                                     
                                            <<<
                                              Context : {context}
                                              Generate the patient's next utterance with the intent: {intent}
                                            >>>
                                            Patient's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return system_prompt_template


def create_message_client_generation_conditionned_da_api(intent_detail_list, intent, context):
    system_prompt_template = f""" 
        <<< Context : {context}
            Generate the therapist's next utterance with the intent: {intent}>>>
        Patient's utterance: """

    messages = [{'role': 'user', 'content': system_prompt_template}]
    return messages


def create_message_therapist_generation_conditionned_da(intent_detail_list, intent, context,theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:

            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" 
    <<< Context : {context}
        Generate the therapist's next utterance with the intent: {intent}>>>
    Therapist's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return messages

def create_message_therapist_generation_conditionned_da_vllm2(intent_detail_list, intent, context,theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:

            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" 
    You are a therapist talking with a patient. Your task is to generate the therapist's next utterance and respect the intent given from one of the following predefined categories : 
- Reflection: The therapist conveys his understanding of what the patient is saying or feeling without judging, interpreting, or advising
- Ask for Information: The therapist seeks further details about a particular event, statement, or background that occurred. 
- Invite to Shift Outlook: The therapist presents a fresh outlook to the patient or prompts him to imagine their response to a situation. 
- Ask about current Emotions: The therapist prompts the patient to recognize and evaluate their current emotions, including identifying the specific emotion and assessing its intensity. 
- Give Solution: The therapist provides direct solutions to address the patient's issues, actively engaging in problem-solving. 
- Planning with the Patient: The therapist collaborates with the patient to develop a concrete action plan. 
- Experience Normalization and Reassurance: The therapist reassures the patient by normalizing their experience, highlighting that others have faced similar situations, fostering a sense of commonality and reassurance. 
- Medical Education and Guidance: The therapist provides the patient with relevant psychological or medical information to enhance understanding and aid in treatment. 
- Greeting or Closing:  The therapist initiates or concludes the conversation with a greeting or closing statement. 
- Backchannel: The therapist acknowledges that they've heard the patient's words by interjecting with a brief response or acknowledgment or repeating exactly what the patient said. ;
- Ask for Consent or Validation: The therapist seeks the patient's consent before posing a question or sharing a statement, or requests confirmation of agreement with the preceding statement 
- Progress Acknowledgment and Encouragement: The therapist recognizes and celebrates the patient's progress or efforts, providing positive reinforcement and encouragement. 
- Empathic Reaction: The therapist openly communicates empathy towards the patient's feelings or circumstances. 
The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.
You will only respond with one therapist's utterance. Do not provide explanations or notes. Make only one proposition. 
The response must be short, no more than 2 or 3 sentences.
    <<< Context : {context}
        Generate the therapist's next utterance with the intent: {intent}>>>
    Therapist's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return system_prompt_template

def create_message_therapist_generation_conditionned_da_vllm2_baseline(intent_detail_list, context,theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:

            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" 
    You are a therapist talking with a patient. Your task is to generate the therapist's next utterance using one of the following intent : 
- Reflection: The therapist conveys his understanding of what the patient is saying or feeling without judging, interpreting, or advising
- Ask for Information: The therapist seeks further details about a particular event, statement, or background that occurred. 
- Invite to Shift Outlook: The therapist presents a fresh outlook to the patient or prompts him to imagine their response to a situation. 
- Ask about current Emotions: The therapist prompts the patient to recognize and evaluate their current emotions, including identifying the specific emotion and assessing its intensity. 
- Give Solution: The therapist provides direct solutions to address the patient's issues, actively engaging in problem-solving. 
- Planning with the Patient: The therapist collaborates with the patient to develop a concrete action plan. 
- Experience Normalization and Reassurance: The therapist reassures the patient by normalizing their experience, highlighting that others have faced similar situations, fostering a sense of commonality and reassurance. 
- Medical Education and Guidance: The therapist provides the patient with relevant psychological or medical information to enhance understanding and aid in treatment. 
- Greeting or Closing:  The therapist initiates or concludes the conversation with a greeting or closing statement. 
- Backchannel: The therapist acknowledges that they've heard the patient's words by interjecting with a brief response or acknowledgment or repeating exactly what the patient said. ;
- Ask for Consent or Validation: The therapist seeks the patient's consent before posing a question or sharing a statement, or requests confirmation of agreement with the preceding statement 
- Progress Acknowledgment and Encouragement: The therapist recognizes and celebrates the patient's progress or efforts, providing positive reinforcement and encouragement. 
- Empathic Reaction: The therapist openly communicates empathy towards the patient's feelings or circumstances. 
The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.
You will only respond with one therapist's utterance. Do not provide explanations or notes. Make only one proposition. 
The response must be short, no more than 2 or 3 sentences.
    <<< Context : {context}
        Generate the therapist's next utterance:>>>
    Therapist's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return system_prompt_template



def create_message_therapist_classification_vllm2_baseline(intent_detail_list,text, context,theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:

            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" 
    Classify the therapist utterance into one of the following intent : 
- Reflection: The therapist conveys his understanding of what the patient is saying or feeling without judging, interpreting, or advising
- Ask for Information: The therapist seeks further details about a particular event, statement, or background that occurred. 
- Invite to Shift Outlook: The therapist presents a fresh outlook to the patient or prompts him to imagine their response to a situation. 
- Ask about current Emotions: The therapist prompts the patient to recognize and evaluate their current emotions, including identifying the specific emotion and assessing its intensity. 
- Give Solution: The therapist provides direct solutions to address the patient's issues, actively engaging in problem-solving. 
- Planning with the Patient: The therapist collaborates with the patient to develop a concrete action plan. 
- Experience Normalization and Reassurance: The therapist reassures the patient by normalizing their experience, highlighting that others have faced similar situations, fostering a sense of commonality and reassurance. 
- Medical Education and Guidance: The therapist provides the patient with relevant psychological or medical information to enhance understanding and aid in treatment. 
- Greeting or Closing:  The therapist initiates or concludes the conversation with a greeting or closing statement. 
- Backchannel: The therapist acknowledges that they've heard the patient's words by interjecting with a brief response or acknowledgment or repeating exactly what the patient said. ;
- Ask for Consent or Validation: The therapist seeks the patient's consent before posing a question or sharing a statement, or requests confirmation of agreement with the preceding statement 
- Progress Acknowledgment and Encouragement: The therapist recognizes and celebrates the patient's progress or efforts, providing positive reinforcement and encouragement. 
- Empathic Reaction: The therapist openly communicates empathy towards the patient's feelings or circumstances. 
Do not provide explanations or notes. Make only one proposition. 
    <<< Context : {context}
        Classify the therapist's next utterance: {text}>>>
    Therapist's utterance Intent: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return system_prompt_template

def create_message_patient_classification_vllm2_baseline(intent_detail_list,text, context,theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:

            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" 
    Classify the patient utterance into one of the following intent : 
-  Sharing personal information or Describe past event: The patient describes a past event (could describe a past unhealthy behavior) or give some personal or medical information. 
- Changing unhealthy behavior in the future: The patient talks about the future and shows an intention or action taken towards actively changing one of his behavior in the future. 
- Sustaining unhealthy behavior in the future: The patient talks about the future and shows an intention or action taken towards actively sustaining one of his behavior in the future or denying one of his behavior. 
- Sharing negative feeling or emotion: The patient describes a negative feeling he has, or the patient describes and explicitly acknowledge a specific negative emotion he is feeling. 
- Sharing positive feeling or emotion: The patient describes a positive feeling he has, or the patient describes and explicitly acknowledge a specific positive emotion he is feeling. 
- Understanding or New Perspective: The patient expresses that he learned or understood something about himself or about his situation or the patient take a new perspective on the situation 
- Greeting or Closing:  The patient initiates or concludes the conversation with a greeting or closing statement. 
- Backchannel: The patient acknowledges that they've heard the therapist's words by interjecting with a brief response or acknowledgment or repeating exactly what the therapist said. 
- Asking for Medical Information: The patient asks for the therapist for medical or therapeutic information 

Do not provide explanations or notes. Make only one proposition. 
    <<< Context : {context}
        Classify the patient's next utterance: {text}>>>
    Patient's utterance Intent: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return system_prompt_template
def create_message_therapist_generation_conditionned_da_api(intent_detail_list, intent, context,theme = None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:

            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    themes = "Your patient is here to talk about"
    if theme == "Smoking":
        themes+="their smoking habbits."
    elif theme == "Drinking":
        themes+="their drinking habbits"
    else:
        themes+="their exercising habbits"
    system_prompt_template = themes+f""" 
    <<< Context : {context}
        Generate the therapist's next utterance with the intent: {intent}>>>
    Therapist's utterance: """

    messages = [{'role': 'user', 'content': system_prompt_template}]
    return messages


def create_message_client_generation_unconditionned(intent_detail_list, n_turn, context):

    intent_example_list = []
    for intent_detail in intent_detail_list:

        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:
            if len(ex) > 3:
                intent_example_list.append(f"{ex}")

    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance \
                                        
                                            ####
                                              Here are some examples:\n {exemples}
                                            ####
                                     The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.\ 
                                     You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition.\
                                     The response must be short, no more than 2 or 3 sentences.\
                                     There is on average 62 turns in a dialog. You are at turn {n_turn}.\
                                     Use [laughter] to signal a patient laughter, [laughs] to signal a patient laugh, [sighs] to signal a patient sigh, [gasps] to signal a patient gasps, [clears throat] to signal a patient clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word.\

                                            <<<
                                            
                                              Context : {context}
                                              Generate the patient's next utterance:
                                            >>>
                                            Patient's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return messages


def create_message_client_generation_conditionned_type(intent_detail_list, type, n_turn, context):

    if 25-int(n_turn) < 12:
        time = "end"
    else:
        time = "begninng"

    if type == "Resistant to change":
        type_def = "You are a patient resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions."
    elif type == "Open to change":
        type_def = "You are a patient open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
    elif type == "Receptive":
        type_def = "You are a patient receptive to therapy, you change you mind during the conversation. "
        if time == "end":
            type_def = type_def + "This is the end of the conversation, You are now more open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
        else:
            type_def = type_def + "This is the beginning of the conversation, You are resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions. "
    intent_example_list = []
    for intent_detail in intent_detail_list:

        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:
            if len(ex) > 3:
                intent_example_list.append(f"{ex}")

    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance \
                                            ####
                                              Here are some examples:\n {exemples}
                                            ####
                                     The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.\ 
                                     You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition.\
                                     The response must be short, no more than 2 or 3 sentences.\
                                     {type_def}\
                                     There is on average 25 turns in a dialog. You are at turn {n_turn}.\
                                     Use [laughter] to signal a patient laughter, [laughs] to signal a patient laugh, [sighs] to signal a patient sigh, [gasps] to signal a patient gasps, [clears throat] to signal a patient clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word.\

                                            <<<
                                              Context : {context}
                                              Generate the patient's next utterance:
                                            >>>
                                            Patient's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return messages

def create_message_client_generation_conditionned_type_api(intent_detail_list, type, context,theme = None):
    themes = "Your are here to talk about"
    if theme == "Smoking":
        themes += "your smoking habbits."
    elif theme == "Drinking":
        themes += "your drinking habbits"
    else:
        themes += "your exercising habbits"

    if "Resistant to change" :
        type_def = "You are a patient resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions."
    elif "Open to change" in type:
        type_def = "You are a patient open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
    elif "Receptive" in type:
        type_def = "You are a patient receptive to therapy, you change you mind during the conversation. "
        if "end" in type:
            type_def = type_def + "This is the end of the conversation, You are now more open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
        else:
            type_def = type_def + "This is the beginning of the conversation, You are resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions. "

    system_prompt_template = themes+f""" 
    Type : {type_def} This is your starting personality but you can change your mind if the therapist builds a good relationship with you and is convincing enough.
    <<< Context : {context}
    Generate the patient 's next utterance and its category in the form""" +"{"+"""Category": [category] "Patient's Utterance": [utterance]} 
    The context represnet the full state of the dialogue.
    If the therapist does not respect basic dialogue rules such as beginning with a greeting or answering questions then add the tag [WRONG] at the begining of the patient utterance.
>>>"""

    messages = [{'role': 'user', 'content': system_prompt_template}]
    return messages

def create_message_client_generation_conditionned_type_vllm(intent_detail_list, type, context, theme=None):

    system_prompt_template =  f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance and respect the intent given from one of the following predefined categories:
-  Sharing personal information or Describe past event: The patient describes a past event (could describe a past unhealthy behavior) or give some personal or medical information. 
- Changing unhealthy behavior in the future: The patient talks about the future and shows an intention or action taken towards actively changing one of his behavior in the future. 
- Sustaining unhealthy behavior in the future: The patient talks about the future and shows an intention or action taken towards actively sustaining one of his behavior in the future or denying one of his behavior. 
- Sharing negative feeling or emotion: The patient describes a negative feeling he has, or the patient describes and explicitly acknowledge a specific negative emotion he is feeling. 
- Sharing positive feeling or emotion: The patient describes a positive feeling he has, or the patient describes and explicitly acknowledge a specific positive emotion he is feeling. 
- Understanding or New Perspective: The patient expresses that he learned or understood something about himself or about his situation or the patient take a new perspective on the situation 
- Greeting or Closing:  The patient initiates or concludes the conversation with a greeting or closing statement. 
- Backchannel: The patient acknowledges that they've heard the therapist's words by interjecting with a brief response or acknowledgment or repeating exactly what the therapist said. 
- Asking for Medical Information: The patient asks for the therapist for medical or therapeutic information 
You are of one of the following type, this type gives your personality:
- Resistant to change : You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions and are really difficult to convince.
- Open to change : You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. You are easy to convince.
-Hesitant : You are a patient receptive to therapy, you change you mind during the conversation. You are in need to take a new perspective to change your mind.
 The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.
You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition. The response must be short, no more than 2 or 3 sentences. Use [laughter] to signal a patient laughter, [laughs] to signal a patient laugh, [sighs] to signal a patient sigh, [gasps] to signal a patient gasps, [clears throat] to signal a patient clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word. If the therapist does not respect basic dialog rules like greeting at the beginning of the conversation or answering questions then add [WRONG] at the beginning of your utterance.
"""
    messages = [{'role': 'system', 'content': system_prompt_template}]
    for i,input in enumerate(exemples_inputs):
        messages += [{'role': 'user', 'content': input}]
        messages += [{'role': 'assistant', 'content': exemples_outputs[i]}]
    themes = "Your are here to talk about"
    if theme == "Smoking":
        themes += "your smoking habbits."
    elif theme == "Drinking":
        themes += "your drinking habbits"
    else:
        themes += "your exercising habbits"

    if "Resistant to change" in type:
        type_def = "You are a patient resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions."
    elif "Open to change" in type:
        type_def = "You are a patient open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
    elif "Receptive" in type:
        type_def = "You are a patient receptive to therapy, you change you mind during the conversation. "
        if "end" in type:
            type_def = type_def + "This is the end of the conversation, You are now more open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
        else:
            type_def = type_def + "This is the beginning of the conversation, You are resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions. "

    system_prompt_template = themes + f""" 
    Type : {type_def} This is your starting personality but you can change your mind if the therapist builds a good relationship with you and is convincing enough.
    <<< Context : {context}
    Generate the patient 's next utterance and its category in the form""" + "{" + """Category": [category] "Patient's Utterance": [utterance]} 
    The context represent the full state of the dialogue.
    If the therapist does not respect basic dialogue rules such as beginning with a greeting or answering questions then add the tag [WRONG] at the begining of the patient utterance.
>>>"""

    messages += [{'role': 'user', 'content': system_prompt_template}]
    return messages
def create_message_therapist_generation_conditionned_da_vllm(intent_detail_list, intent, context, theme=None):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:

            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)

    system_prompt_template = f"""You are a therapist talking with a patient. Your task is to generate the therapist's next utterance and respect the intent given from one of the following predefined categories : 
- Reflection: The therapist conveys his understanding of what the patient is saying or feeling without judging, interpreting, or advising
- Ask for Information: The therapist seeks further details about a particular event, statement, or background that occurred. 
- Invite to Shift Outlook: The therapist presents a fresh outlook to the patient or prompts him to imagine their response to a situation. 
- Ask about current Emotions: The therapist prompts the patient to recognize and evaluate their current emotions, including identifying the specific emotion and assessing its intensity. 
- Give Solution: The therapist provides direct solutions to address the patient's issues, actively engaging in problem-solving. 
- Planning with the Patient: The therapist collaborates with the patient to develop a concrete action plan. 
- Experience Normalization and Reassurance: The therapist reassures the patient by normalizing their experience, highlighting that others have faced similar situations, fostering a sense of commonality and reassurance. 
- Medical Education and Guidance: The therapist provides the patient with relevant psychological or medical information to enhance understanding and aid in treatment. 
- Greeting or Closing:  The therapist initiates or concludes the conversation with a greeting or closing statement. 
- Backchannel: The therapist acknowledges that they've heard the patient's words by interjecting with a brief response or acknowledgment or repeating exactly what the patient said. ;
- Ask for Consent or Validation: The therapist seeks the patient's consent before posing a question or sharing a statement, or requests confirmation of agreement with the preceding statement 
- Progress Acknowledgment and Encouragement: The therapist recognizes and celebrates the patient's progress or efforts, providing positive reinforcement and encouragement. 
- Empathic Reaction: The therapist openly communicates empathy towards the patient's feelings or circumstances. 
The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.
You will only respond with one therapist's utterance. Do not provide explanations or notes. Make only one proposition. 
The response must be short, no more than 2 or 3 sentences.
### Here are some esxemples :
{exemples}
###
Use [laughter] to signal a therapist laughter, [laughs] to signal a therapist laugh, [sighs] to signal a therapist sigh, [gasps] to signal a therapist  gasps, [clears throat] to signal a therapist clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word. """
    themes = "Your patient is here to talk about"
    if theme == "Smoking":
        themes += "their smoking habbits."
    elif theme == "Drinking":
        themes += "their drinking habbits"
    else:
        themes += "their exercising habbits"
    prompt_template = themes + f""" 
    <<< Context : {context}
        Generate the therapist's next utterance with the intent: {intent}>>>
    Therapist's utterance: """
    messages = [{'role': 'system', 'content': system_prompt_template}]
    messages += [{'role': 'user', 'content': prompt_template}]
    return messages
