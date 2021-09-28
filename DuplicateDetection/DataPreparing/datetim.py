'''
from datetime import datetime
#x=float(input())
#a=[2009,11,12]
print(datetime.fromtimestamp(947084781))
print(datetime.timestamp(datetime(1999,9,17)))
'''

from transformers import MBartTokenizer, MBartForConditionalGeneration

article_text = ''' Создание чат-бота/помощника, который будет отвечать на типичные вопросы сотрудников, например, "как сделать скан трудовой", "где мне найти пример какого-нибудь договора", "как уйти в админ?", "как мне уйти в декрет, если я мужчина, какие мне нужны документы", и еще миллион подобоного рода вопросов, адресованных в HR, Adms, ФЮС, T&D. Мое предложение - содать бота, который будет, например, по запросу "трудовая книжка" - сразу же выдавать ссылку на портал, где все подробно описано и указаны правильные контактные лица (техническую реализацию можно и нужно еще обсуждать, как будет максимально юзер-френдли).'''
model_name = "IlyaGusev/mbart_ru_sum_gazeta"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

input_ids = tokenizer.prepare_seq2seq_batch(
    [article_text],
    src_lang="en_XX", # fairseq training artifact
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=600
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=162,
    no_repeat_ngram_size=3,
    num_beams=5,
    top_k=0
)[0]

summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(summary)



