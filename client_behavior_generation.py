import json
import argparse
from tqdm import tqdm
import codecs
from vllm import LLM, SamplingParams
from utils_mistral import *

intent_detail_list = read_prompt_csv('client')

def generate_client_intent(context,condition,intent='None',n_turn=0,type='None'):

	if condition == 'DA':
		messages = create_message_client_generation_conditionned_da(intent_detail_list,intent,context)
	elif condition == 'Type':
		messages = create_message_client_generation_conditionned_type(intent_detail_list,type,n_turn,context)
	elif condition == 'Unconditionned':
		messages = create_message_client_generation_unconditionned(intent_detail_list,n_turn,context)
	else:
		print('Error: condition not recognized')
		return None
	response = get_completion_from_messages_local(messages, temperature=0.7)

	return response

def generate_client_intent_api(context,condition,intent='None',n_turn=0,type='None',theme='None'):
	agent_id = "ag:90175714:20241126:untitled-agent:203cba4b"
	API = "q40QH3o9XHh5Vr98HmEp35uDacaYIUUu"

	sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
	if condition == 'DA':
		messages = create_message_client_generation_conditionned_da_api(intent_detail_list,intent,context)
	elif condition == 'Type':
		messages = create_message_client_generation_conditionned_type_api(intent_detail_list,type,context,theme=theme)
	elif condition == 'Unconditionned':
		messages = create_message_client_generation_unconditionned(intent_detail_list,n_turn,context)
	else:
		print('Error: condition not recognized')
		return None
	response = get_completion_from_messages_api(messages,agent_id, API,temperature=0.7)

	return response

def generate_client_intent_vllm(llm,context,condition,intent='None',n_turn=0,type='None',theme='None'):
	agent_id = "ag:90175714:20241126:untitled-agent:203cba4b"
	API = "q40QH3o9XHh5Vr98HmEp35uDacaYIUUu"

	sampling_params = SamplingParams(temperature=0.3, top_p=0.95,max_tokens=8192)
	if condition == 'DA':
		prompts = create_message_client_generation_conditionned_da_vllm(intent_detail_list, intent, context, theme=theme)
		#messages = create_message_client_generation_conditionned_da(intent_detail_list,intent,context,theme=theme)
	elif condition == 'Type':
		messages = create_message_client_generation_conditionned_type_vllm(intent_detail_list,type,context,theme=theme)
	elif condition == 'Unconditionned':
		messages = create_message_client_generation_unconditionned(intent_detail_list,n_turn,context)
	else:
		print('Error: condition not recognized')
		return None

	outputs =llm.generate(prompts, sampling_params,use_tqdm=False)
	return outputs[0].outputs[0].text

def generate_client_intent_vllm_parra(llm,context,condition,intent='None',n_turn=0,type='None',theme='None'):
	agent_id = "ag:90175714:20241126:untitled-agent:203cba4b"
	API = "q40QH3o9XHh5Vr98HmEp35uDacaYIUUu"

	sampling_params = SamplingParams(temperature=0.3, top_p=0.95,max_tokens=8192)
	if condition == 'DA':
		prompts = [create_message_client_generation_conditionned_da_vllm(intent_detail_list, intent[i], context[i], theme=theme[i]) for i in range(len(intent))]
		#messages = create_message_client_generation_conditionned_da(intent_detail_list,intent,context,theme=theme)
	elif condition == 'Type':
		prompts = [create_message_client_generation_conditionned_type_vllm(intent_detail_list,type,context,theme=theme) for i in range(len(intent))]
		#messages = create_message_client_generation_conditionned_type_vllm(intent_detail_list,type,context,theme=theme)
	elif condition == 'Unconditionned':
		messages = create_message_client_generation_unconditionned(intent_detail_list,n_turn,context)
	else:
		print('Error: condition not recognized')
		return None

	outputs =llm.generate(prompts, sampling_params)
	responses = []
	for output in outputs:
		responses.append(output.outputs[0].text)
	return responses

def generate_client_intent_vllm_parra_baseline(llm, context, condition, intent='None', n_turn=0, type='None', theme='None'):
	agent_id = "ag:eb6ed31c:20241120:conditionned-mi-therapist:172de02b"
	API = "hVnDmp9N4SAnKpTm5YwcEBFX18xbFH3A"
	sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=8192)
	intent_list = [intent_detail_list[i]['intent'] for i in range(len(intent_detail_list))]
	if condition == 'DA':
		prompts = [create_message_therapist_generation_conditionned_da_vllm2_baseline(intent_detail_list, context[i],theme=theme[i]) for i in range(5)]
	elif condition == 'Type':
		prompts = [create_message_client_generation_conditionned_type_vllm(intent_detail_list, type, context, theme=theme) for i in range(5)]
	else:
		print('Error: condition not recognized')
		return None

	outputs = llm.generate(prompts, sampling_params)
	responses = []
	for output in outputs:
		responses.append(output.outputs[0].text)
	classification_prompts = [create_message_patient_classification_vllm2_baseline(intent_detail_list,responses[i], context[i],theme=theme[i]) for i in range(5)]
	outputs = llm.generate(classification_prompts, sampling_params)
	actions = []
	for output in outputs:
		action = 'Unknown'

		for intent in intent_list:
			if intent in output.outputs[0].text:
				action = intent
		actions.append(action)
	print(actions)
	return responses,actions