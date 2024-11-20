from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.1")



print(llm("Me conta uma história sobre IA. E pessoas importantes nesse cenário"))




