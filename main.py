import json

from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util


# Configuração do modelo LlamaCpp
def setup_llama_model(model_path, n_ctx=2048, n_gpu_layers=20):
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )
    return llm

# Carregar perguntas e respostas do arquivo JSON
def load_qa_data(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as file:
        qa_data = json.load(file)
    return qa_data

# Configurar a cadeia de perguntas e respostas
def setup_qa_chain(llm):
    template = """You are a helpful assistant with knowledge based on the provided QA data.

Question: {question}

Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain

# Configurar o modelo de embeddings
def setup_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Buscar resposta baseada em similaridade semântica
def search_similar_question(question, qa_data, embedding_model, threshold=0.75):
    # Extrair perguntas do JSON
    json_questions = [item["question"] for item in qa_data]
    json_embeddings = embedding_model.encode(json_questions, convert_to_tensor=True)
    
    # Embedding da pergunta do usuário
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    
    # Calcular similaridade
    similarities = util.pytorch_cos_sim(question_embedding, json_embeddings)
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[0, best_match_idx].item()
    
    # Se a similaridade for alta, retornar a resposta correspondente
    if best_match_score >= threshold:
        return qa_data[best_match_idx]["answer"], qa_data[best_match_idx]["question"]
    return None, None

# Função principal
def main():
    # Configurações
    model_path = "./local-llama-langchain/models/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf"  # Caminho para o modelo
    json_file_path = "qa.json"  # Caminho para o arquivo JSON

    # Carregar modelo e dados
    print("Carregando o modelo LlamaCpp...")
    llm = setup_llama_model(model_path)
    
    print("Carregando dados de QA do JSON...")
    qa_data = load_qa_data(json_file_path)

    print("Carregando modelo de embeddings...")
    embedding_model = setup_embedding_model()

    # Configurar cadeia de QA
    print("Configurando a cadeia de perguntas e respostas...")
    qa_chain = setup_qa_chain(llm)

    # Loop de interação
    while True:
        question = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        if question.strip().lower() == "sair":
            print("Encerrando...")
            break

        # Procurar resposta com base em similaridade
        print("\nProcurando perguntas semelhantes no JSON...")
        answer, matched_question = search_similar_question(question, qa_data, embedding_model)
        
        if answer:
            print(f"\nPergunta semelhante encontrada no JSON: \"{matched_question}\"")
            print(f"Resposta direta do JSON: {answer}")
        else:
            # Responder com o modelo
            print("\nBuscando resposta com o modelo...")
            result = qa_chain.run(question)
            print(f"\nResposta do modelo: {result}")

if __name__ == "__main__":
    main()
