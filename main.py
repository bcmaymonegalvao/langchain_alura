from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo:str = Field("motivo pelo qual é interessante visitar essa cidade")

class Restaurantes(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurantes:str = Field("Restaurante recomendados na cidade")    

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado  o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}"
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

cadeia_1 = prompt_cidade | modelo | parseador_destino
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

resposta = cadeia.invoke(
    {
        "interesse" : "praias"
    }
)

print(resposta)


# prompt = modelo_de_prompt.format(
#     dias=numero_dias,
#     numero_criancas = numero_criancas,
#     atividade=atividade
# )

# prompt = f"Crie um roteiro de viagens, para um período de {numero_dias} dias, para uma família com {numero_criancas} crianças que busca atividades relacionadas a {atividade}."

# prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma familia com {numero_criancas} crianças, que gosta de {atividade}."

# cliente = OpenAI(api_key=api_key)

# resposta = cliente.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "system",
#             "content": "Você é um assistente de roteiro de viagens."
#         },
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
# )

# resposta_em_texto = resposta.choices[0].message.content
# print(resposta_em_texto)