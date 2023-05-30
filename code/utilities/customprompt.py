# flake8: noqa
from langchain.prompts import PromptTemplate

template = """Tenendo conto di questi esempi di categorizzazione di email:
{summaries}
 
Categorizza il seguente esempio di mail: 
{question}
Scegliento tra le seguenti categorie:
anomalia_login
carta_smagnetizzata
carta_smarrimento
carta_spedizione
chiusura_conto
Contest e Promo
Disputa
EC
Hype_limiti_start
password_modifica
ricarica
stato_pratica

Includi i riferimenti alle risorse se sono rilevanti ("SOURCES").
Risposta:"""

PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)


