import time

import numpy as np
from openai import OpenAI
import torch

import json
from sentence_transformers import SentenceTransformer
from client_behavior_generation import generate_client_intent_vllm_parra, generate_client_intent_vllm_parra_baseline
from therapist_behavior_generation import generate_therapist_intent_vllm_parra,generate_therapist_intent_vllm_parra_baseline
class UserMI:

    def __init__(self,n_parra, type=None):
        self.n_parra = n_parra
        self.action_space = \
            ["SharingpersonalinformationorDescribepastevent",
             "Changingunhealthybehaviorinthefuture",
             "Sustainingunhealthybehaviorinthefuture",
             "Sharingnegativefeelingoremotion",
             "Sharingpositivefeelingoremotion",
             "UnderstandingorNewPerspective",
             "GreetingorClosing",
             "Backchannel",'Unknown',"AskingforMedicalInformation"]
        self.agent_action_space = \
            ["Reflection",
             "Ask for Information",
             "Invite to Shift Outlook",
             "Ask about current Emotions",
             "Give Solution",
             "Planning with the Patient",
             "Experience Normalization and Reassurance",
             "Medical Education and Guidance",
             "Greeting or Closing",
             "Backchannel",
             "Ask for Consent or Validation",
             "Progress Acknowledgment and Encouragement",
             "Empathic Reaction"]
        self.termination_da = 8
        self.action_to_id = {}
        for i, a in enumerate(self.action_space):
            self.action_to_id[a] = i

        self.agent_action_to_id = {}
        for i, a in enumerate(self.agent_action_space):
            self.agent_action_to_id[a] = i

        self.last_action = [0 for i in range(self.n_parra)]
        self.current_action = [0 for i in range(self.n_parra)]
        self.theme = [np.random.randint(3) for i in range(self.n_parra)]
        self.themes = ['Smoking', 'Drinking', 'Exercice']
        self.types_to_id = {'Receptive': 0, 'Resistant to change': 1, 'Open to change': 2}
        self.id_to_type = ['Receptive', 'Resistant to change', 'Open to change']
        if type is not None:
            self.type = [type for i in range(self.n_parra)]
        else:
            self.type = [np.random.randint(3) for i in range(self.n_parra)]
        self.random_seed = 0
        self.n_action = len(self.action_space)
        self.past_value_da = np.zeros((n_parra,23))
        self.past_time_features = np.zeros((n_parra,23))
        self.past_observed_mask = np.zeros((n_parra,23))
        self.rapport = [0 for i in range(self.n_parra)]
        self.context = [0 for i in range(self.n_parra)]
        self.perspective = [0 for i in range(self.n_parra)]
        self.last_therapist_text = ["" for i in range(self.n_parra)]
        self.therapist_text = ["" for i in range(self.n_parra)]
        self.last_patient_text = ["" for i in range(self.n_parra)]
        self.patient_text = ["" for i in range(self.n_parra)]
        self.text_context = ["Context: " for i in range(self.n_parra)]
        self.turn = [0 for i in range(self.n_parra)]
        self.cluster = True

        self.encoded_text = torch.zeros((n_parra,768))
        if self.cluster:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model="/home/galland/mistral_models/NemoInstruct", max_model_len=5000,
                       tokenizer_mode="mistral", load_format="mistral", config_format="mistral")
            self.embedding_model = SentenceTransformer("/home/galland/mistral_models/nomic",trust_remote_code=True)
        else:
            self.embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",trust_remote_code=True)

    def create_context_i(self,i, turn):
        if self.last_therapist_text[i] == "":
            if self.last_patient_text[i] == "":
                if self.therapist_text[i] == "":
                    return ""
                else:
                    return "Context turn " + str(turn) + ": Therapist : " + self.therapist_text[i]
            else:
                return "Context turn " + str(
                    turn) + ": Patient : " + self.last_patient_text[i] + " Therapist : " + self.therapist_text[i]
        else:
            return "Context turn " + str(
                turn) + ": Therapist : " + self.last_therapist_text[i] + " Patient : " + self.patient_text[i] + " Therapist : " + self.therapist_text[i]
    def create_context(self, turn):
        res = []
        for i in range(self.n_parra):
            res.append(self.create_context_i(i,turn))
        return res
    def create_context_therapist_i(self,i, turn):
        if self.last_therapist_text[i] == "":
            if self.last_patient_text[i] == "":
                if self.patient_text[i] == "":
                    return ""
                else:
                    return "Context turn " + str(turn) + ": Patient : " + self.patient_text[i]
            else:
                return "Context turn " + str(
                    turn) + ": Therapist : " + self.last_therapist_text[i] + " Patient : " + self.patient_text[i]
        else:
            return "Context turn " + str(
                turn) + ": Patient : " + self.last_patient_text[i] + " Therapist : " + self.therapist_text[i] + " Patient : " + self.patient_text[i]
    def create_context_therapist(self, turn):
        res = []
        for i in range(self.n_parra):
            res.append(self.create_context_therapist_i(i,turn))
        return res

    def react(self, agent_da, turn_id,baseline=False):
        self.last_therapist_text = self.therapist_text
        self.turn = [self.turn[i] +1 for i in range(self.n_parra)]
        error = True
        essai = 0
        if baseline:
            self.therapist_text,agent_da = generate_therapist_intent_vllm_parra_baseline(self.llm, self.text_context,
                                                                 'DA',
                                                                 intent=[self.agent_action_space[agent_da[i]] for i in range(self.n_parra)] ,
                                                                 theme=[self.themes[self.theme[i]] for i in range(self.n_parra)] )
            agent_da = [self.agent_action_to_id[a.replace(' ','').replace('"','')] for a in agent_da]
        else:
            self.therapist_text = generate_therapist_intent_vllm_parra(self.llm, self.text_context,
                                                                 'DA',
                                                                 intent=[self.agent_action_space[agent_da[i]] for i in range(self.n_parra)] ,
                                                                 theme=[self.themes[self.theme[i]] for i in range(self.n_parra)] )


        for i in range(self.n_parra):
            self.text_context[i] += "Therapist: "
            self.text_context[i]+=self.therapist_text[i]
            self.past_value_da[i]= np.concatenate((self.past_value_da[i,1:],[agent_da[i] + self.n_action + 2]))
            self.past_time_features[i]=np.concatenate((self.past_time_features[i,1:],[turn_id[i]]))
            self.past_observed_mask[i]=np.concatenate((self.past_observed_mask[i,1:],[1]))
        start = time.time()

        start = time.time()

        error = True
        essai = 0
        type = [self.id_to_type[self.type[i]] for i in range(self.n_parra)]
        for i in range(self.n_parra):
            if self.type[i] == 0:
                if turn_id[i]  < 20:
                    type[i] += ' beginning of the dialogue'
                else:
                    type[i] += " end of the dialogue"
        if self.cluster:
            text,action =generate_client_intent_vllm_parra_baseline(self.llm, self.text_context, 'DA',
                                               type=type,
                                               theme=[self.themes[self.theme[i]] for i in range(self.n_parra)] )

            for i in range(self.n_parra):
                self.patient_text[i] = text[i]
                action[i] = self.action_to_id[action[i].replace(' ','')]

        for i in range(self.n_parra):
            self.text_context[i] += "Patient: "
            self.text_context[i] += self.patient_text[i]
            self.past_observed_mask[i] = np.concatenate((self.past_observed_mask[i,1:], [1]))
            self.past_value_da[i] = np.concatenate((self.past_value_da[i,1:], [action[i]]))
            self.past_time_features[i] = np.concatenate((self.past_time_features[i,1:], [turn_id[i] + 1]))

        self.last_action = self.current_action
        self.current_action = action
        self.last_patient_text = self.patient_text

        return action

    def get_reward(self, action):
        r = np.zeros(self.n_parra)

        for i in range(self.n_parra):
            if "[WRONG]" in self.patient_text:
                r[i] -= 1
            if self.action_space[action[i] ] == "Sharingpositivefeelingoremotion":
                self.rapport[i] += 1
                if self.rapport[i] <=3:
                    r[i] += 50
            elif self.action_space[action[i] ] == "UnderstandingorNewPerspective":
                r[i] += 0
                if self.rapport[i] > 2 and self.context[i] > 2 and self.perspective[i] <= 3:
                    r[i] += 150
                    self.perspective[i] += 1
            elif self.action_space[action[i] ] == "GreetingorClosing":
                r[i] += 0
            elif self.action_space[action[i] ] == "AskingforMedicalInformation":
                if self.rapport[i] >2 and self.context[i] > 2 and self.perspective[i] > 2:
                    r[i] += 245
                elif self.rapport[i] > 2 and self.context[i] <= 3:
                    r[i] += 100
                    self.context[i] += 1
            elif self.action_space[action[i] ] == "Sharingnegativefeelingoremotion":
                self.rapport[i] += 1
                if self.rapport[i] <= 3:
                    r[i] += 50
            elif self.action_space[action[i] ] == "Changingunhealthybehaviorinthefuture":
                r[i] += 5
                if self.rapport[i] >2 and self.context[i] > 2 and self.perspective[i] > 2:
                    r[i] += 245
                elif self.rapport[i] > 2 and self.context[i] > 2 and self.perspective[i] <= 3:
                    r[i] += 150
                    self.perspective[i] += 1
            elif self.action_space[action[i] ] == "Sustainingunhealthybehaviorinthefuture":
                r[i] -= 10
            elif self.action_space[action[i] ] == "SharingpersonalinformationorDescribepastevent":
                r[i] += 0
                if self.rapport[i] > 2 and self.context[i] <= 3:
                    r[i] += 100
                    self.context[i] += 1

        return r

        return r

    def reset(self, seed=None):
        self.last_action = [0 for i in range(self.n_parra)]
        self.current_action = [0 for i in range(self.n_parra)]
        self.random_seed = seed
        self.past_value_da = np.zeros((self.n_parra,23))
        self.past_time_features = np.zeros((self.n_parra,23))
        self.past_observed_mask = np.zeros((self.n_parra,23))
        self.rapport = [0 for i in range(self.n_parra)]
        self.context = [0 for i in range(self.n_parra)]
        self.text_context = ["Context: " for i in range(self.n_parra)]
        self.perspective = [0 for i in range(self.n_parra)]
        self.last_therapist_text = ["" for i in range(self.n_parra)]
        self.therapist_text =   ["" for i in range(self.n_parra)]
        self.last_patient_text = ["" for i in range(self.n_parra)]
        self.patient_text = ["" for i in range(self.n_parra)]
        self.theme = [np.random.randint(0,3) for i in range(self.n_parra)]

    def reset_i(self, i):
        self.last_action[i] = 0
        self.current_action[i] = 0
        self.past_value_da[i] = np.zeros(23)
        self.past_time_features[i] = np.zeros(23)
        self.past_observed_mask[i] = np.zeros(23)
        self.rapport[i] = 0
        self.context[i] = 0
        self.text_context[i] = "Context: "
        self.perspective[i] = 0
        self.last_therapist_text[i] = ""
        self.therapist_text[i] = ""
        self.last_patient_text[i] = ""
        self.patient_text[i] = ""
        self.theme[i] = np.random.randint(0,3)
    def set_type(self, type):
        self.type = [type for i in range(self.n_parra)]

    def seed(self, seed):
        self.random_seed = seed
