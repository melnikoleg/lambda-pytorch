import torch
import random
import json
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel

device = "cpu"
model = GPT2LMHeadModel.from_pretrained("model/").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("model/", use_fast=True)

def ans(question, description=''):
    seed = random.randint(1, 10000000)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inp = tokenizer.encode(f'Вопрос: {question}\nОписание: {description}\nОтвет:',return_tensors="pt").to(device)
    gen = model.generate(inp, do_sample=True, top_p=0.9, temperature=0.86, max_new_tokens=100, repetition_penalty=1.2) #, stop_token="<eos>")
    
    gen = tokenizer.decode(gen[0])
    return gen[:gen.index('<eos>') if '<eos>' in gen else len(gen)]


def lambda_handler(event, context):

    body = json.loads(event['body'])

    question = body['question']
    context = body['context']

    answer = ans(question=question)

    print('Question: {0}, Answer: {1}'.format(question, answer))

    return {
        'statusCode': 200,
        'body': json.dumps({
            'Question': question,
            'Answer': answer
        })
    }