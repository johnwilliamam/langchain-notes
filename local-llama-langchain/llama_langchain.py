from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import \
    StreamingStdOutCallbackHandler  # for streaming resposne
from langchain.llms import LlamaCpp, OpenAI

# Make sure the model path is correct for your system!
model_path = "./models/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf" 
# <-------- enter your model path here 


template = """Question: {question}

Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# llm = LlamaCpp(
#     model_path=model_path,
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     callback_manager=callback_manager,
#     verbose=True,
#     # temperature=1
# )

# Uncomment the code below if you want to run inference on CPU
llm = LlamaCpp(
    temperature=0.5,
    model_path=model_path,
    n_ctx=10000,
    n_gpu_layers=8,
    n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=512,
    repeat_penalty=1.5,
    top_p=0.5,
    verbose=True,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Me conte uma piada "


print(llm_chain.run(question))
