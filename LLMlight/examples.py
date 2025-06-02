# %%
# import LLMlight as llm
# print(dir(llm))
# print(llm.__version__)
from LLMlight import LLMlight

# Initialize with default settings
# model = LLMlight()

# Run a simple query
# response = model.prompt('What is the capital of France?', system="You are a helpful assistant.")

#%%
system = """Je bent een Nederlandse AI-assistent gespecialiseerd in het omzetten van
transcripties naar gestructureerde en overzichtelijke notulen. Jouw taak is om van een
transcriptie een professioneel verslag te maken, zelfs als de transcriptie afkomstig is
van automatische spraak-naar-tekst software en fouten kan bevatten. Je mag aannames maken
indien het de kwaliteit van de output zal verbeteren.
"""

query = """Je ontvangt een transcriptie van de gebruiker als input. Zet deze direct om in volledig
gestructureerde en gepolijste notulen volgens de bovenstaande richtlijnen.
Wanneer je klaar bent, geef je alleen het uiteindelijke verslag als output, zonder verdere uitleg.
"""

instructions = """Bij het verwerken van de transcriptie, houd je rekening met het volgende:
    1. **Corrigeren van fouten:** Je corrigeert duidelijke fouten in de transcriptie (zoals
    verkeerde woorden, grammaticale fouten en onduidelijke zinnen) op basis van de context.
    Als iets onzeker blijft, markeer je dit met '[?]'.
    2. **Heldere structuur:** Je formatteert de notulen volgens de volgende opbouw:
       - **Titel en datum van de bijeenkomst** (haal dit uit de context van de
       transcriptie, indien mogelijk, anders laat het leeg).
       - **Aanwezigen en afwezigen** (indien vermeld).
       - **Samenvatting:** Een beknopte samenvatting van de belangrijkste besproken
       onderwerpen en uitkomsten.
       - **Details per agendapunt:** Geef de belangrijkste punten en discussies weer per
       onderwerp.
       - **Actiepunten en besluiten:** Noteer actiepunten en besluiten genummerd en
       duidelijk geordend.
    3. **Samenvatten en structureren:** Behoud de kern van de informatie, verwijder
    irrelevante details en vermijd herhaling. Gebruik bondige, professionele taal.
    4. **Neutraliteit:** Schrijf in een objectieve, neutrale toon en geef geen subjectieve
    interpretaties.
    5. **Tijdsaanduidingen:** Voeg waar nodig tijdsaanduidingen toe om de volgorde van de
    bespreking te verduidelijken. Laat irrelevante tijdsaanduidingen weg.
    6. De context is in het Nederlands en de output zal jij ook schrijven in het Nederlands.
    """


from LLMlight import LLMlight
modelname = 'deepseek-r1-0528-qwen3-8b'
modelname = 'hermes-3-llama-3.2-3b'

preprocessing='global-reasoning',
preprocessing='chunk-wise'

model = LLMlight(modelname=modelname,
                 preprocessing=preprocessing,
                 method=None,
                 temperature=0.8,
                 top_p=1,
                 chunks={'type': 'chars', 'size': 8192, 'overlap': 2000},
                 n_ctx=16384,
                 verbose='debug',
                 )

# Run model
response = model.prompt(query,
                   instructions=instructions,
                   context=context,
                   system=system,
                   stream=False,
                   )
print(response)


#%%
# Run model
response2 = model.global_reasoning(query,
                   context=context,
                   instructions=instructions,
                   system=system,
                   return_per_chunk=False,
                   stream=False,
                   )
print(response2)

# Run model
response3 = model.chunk_wise(query,
                   context=context,
                   instructions=instructions,
                   system=system,
                   return_per_chunk=False,
                   stream=False,
                   )
print(response3)

# %%
from LLMlight import LLMlight
model = LLMlight(verbose='debug')
model.check_logger()


#%% Available models
from LLMlight import LLMlight
model = LLMlight(verbose='info')
modelnames = model.get_available_models(validate=False)

# %%
for modelname in modelnames:
    from LLMlight import LLMlight
    llm = LLMlight(modelname=modelname)
    print(llm.modelname)

    system_message = "You are a helpful assistant."
    response = llm.prompt('What is the capital of France?', system=system_message)
    print(response)

# %%
from LLMlight import LLMlight
model_path = r'C:/Users\beeld/.lmstudio/models/NousResearch/Hermes-3-Llama-3.2-3B-GGUF\Hermes-3-Llama-3.2-3B.Q4_K_M.gguf'
model = LLMlight(endpoint=model_path, top_p=0.9)
# model.prompt('hello, who are you?')
system_message = "You are a helpful assistant."
response = model.prompt('What is the capital of France?', system=system_message)

#%%
from LLMlight import LLMlight

# Initialize model
model = LLMlight()

# Read and process PDF
model.read_pdf(r'D://OneDrive - Tilburg University//TiU//Introduction new colleagues.pdf')

# Query about the document
response = model.prompt('Summarize the main points of this document', global_reasoning=True)

print(response)

#%%
import llama_cpp
print(llama_cpp.__version__)
print(llama_cpp.llama_cpp_version())  # Might crash if incompatible

# Check your GGUF model's metadata
model_path = r'C:/Users\beeld/.lmstudio/models/NousResearch/Hermes-3-Llama-3.2-3B-GGUF\Hermes-3-Llama-3.2-3B.Q4_K_M.gguf'
with open(model_path, 'rb') as f:
    header = f.read(128)
    print(header)


#%%
import os
import logging
import requests
from tqdm import tqdm
from llama_cpp import Llama

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def download_and_load_gguf_model(
    url: str,
    model_name: str,
    cache_dir: str = "local_models",
    n_ctx: int = 4096,
    n_threads: int = 8,
    n_gpu_layers: int = 0,
    verbose: bool = True
) -> Llama:
    """
    Downloads a GGUF model from a URL (if not already cached) and loads it with llama-cpp-python.

    Args:
        url (str): Direct URL to the .gguf model file.
        model_name (str): Filename to use for local caching (e.g. 'Hermes.gguf').
        cache_dir (str): Directory to store the model. Default is 'local_models'.
        n_ctx (int): Context window size.
        n_threads (int): CPU threads to use.
        n_gpu_layers (int): GPU layers to offload. Use 0 for CPU-only.
        verbose (bool): Print logs during loading.

    Returns:
        Llama: Loaded model ready for inference.
    """
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, model_name)

    if not os.path.exists(model_path):
        logger.info(f"Model not found locally. Downloading from:\n{url}")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(model_path, "wb") as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=model_name
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    else:
        logger.info(f"Using cached model at: {model_path}")

    # Load with llama-cpp
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose
    )

    logger.info("Model loaded successfully.")
    return llm

#%%
url = "https://huggingface.co/TheBloke/Hermes-2-Pro-Llama-3-GGUF/resolve/main/hermes-2-pro-llama-3.Q4_K_M.gguf"

model_name = "hermes-2-pro-llama-3.Q4_K_M.gguf"

llm = download_and_load_gguf_model(url, model_name)

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhat is the capital of France?\n<|start_header_id|>assistant<|end_header_id|>\n"
response = llm(prompt=prompt, max_tokens=20, stop=["<|end_of_text|>"])
print(response["choices"][0]["text"].strip())

#%%
# Code to inference Hermes with HF Transformers
# Requires pytorch, transformers, bitsandbytes, sentencepiece, protobuf, and flash-attn packages

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import bitsandbytes, flash_attn

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-3-Llama-3.1-8B', trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Hermes-3-Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True
)

prompts = [
    """<|im_start|>system
You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
<|im_start|>user
Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world.<|im_end|>
<|im_start|>assistant""",
    ]

for chat in prompts:
    print(chat)
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Response: {response}")

#%%
url = "https://huggingface.co/TheBloke/Hermes-2-Pro-Llama-3-GGUF/resolve/main/hermes-2-pro-llama-3.Q4_K_M.gguf"
model_name = "hermes-2-pro-llama-3.Q4_K_M.gguf"

# Already avoids bitsandbytes
llm = download_and_load_gguf_model(url, model_name, n_gpu_layers=0)



from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"  # for example

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")  # will crash without CUDA if quantized
